import json
from pathlib import Path

import modal

app = modal.App("decido-vision")

# Persistent volume caches model weights across cold starts — without this,
# every cold start re-downloads ~15GB from HuggingFace.
volume = modal.Volume.from_name("decido-model-weights", create_if_missing=True)
MODEL_DIR = Path("/models/qwen2.5-vl")
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.49",
        "accelerate>=0.30",
        "qwen-vl-utils>=0.0.8",
        "Pillow>=10.0",
    )
)

_SYSTEM_PROMPT = """\
You are a browser automation agent. Given a screenshot and a task, propose up \
to 3 UI actions that would best complete the task.

Respond ONLY with a JSON array. Each element must have:
  - action     (one of: click, type, scroll, select, hover)
  - bbox       ([x1, y1, x2, y2] in pixel coordinates)
  - confidence (float 0–1, reflecting how certain you are this action advances the task;
                vary this based on element clarity, label match, and context)
  - text       (string, required for "type" and "select", omit otherwise)

Example (confidence values are illustrative — set yours based on actual certainty):
[
  {"action": "click", "bbox": [120, 340, 200, 360], "confidence": 0.94},
  {"action": "type",  "bbox": [100, 200, 300, 230], "confidence": 0.73, "text": "hello@example.com"},
  {"action": "click", "bbox": [340, 510, 420, 535], "confidence": 0.61}
]
"""


@app.cls(
    image=image,
    gpu="A10G",
    volumes={str(MODEL_DIR.parent): volume},
    timeout=120,
    scaledown_window=300,   # keep warm for 5 min between requests
)
class VisionModel:

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Download weights on first cold start, then served from volume
        if not (MODEL_DIR / "config.json").exists():
            from huggingface_hub import snapshot_download
            snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR))
            volume.commit()

        self.processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    @modal.method()
    def propose(self, screenshot_bytes: bytes, task: str, memory_context: str | None = None) -> list[dict]:
        """
        Run Qwen2.5-VL on a screenshot and return raw action proposals as dicts.

        Returns an empty list on any inference or parse failure — callers should
        treat an empty list as a degraded (DOM-only) run, not a hard error.
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        import io

        try:
            image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")

            messages = [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Task: {task}" + (f"\n\n{memory_context}" if memory_context else "")},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to("cuda")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                )

            # Decode only the newly generated tokens (strip the input prompt)
            new_tokens = output_ids[0][len(inputs.input_ids[0]):]
            response = self.processor.decode(new_tokens, skip_special_tokens=True)

            return _parse_response(response)

        except Exception:
            return []


def _parse_response(text: str) -> list[dict]:
    """Extract the JSON array from the model response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []
