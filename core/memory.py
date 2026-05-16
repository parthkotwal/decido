"""
Episodic memory for multi-step sessions.

After each step, the page state + action + outcome is embedded and stored.
At the start of each new step, the current page state + task is embedded
and the top-k most similar past steps are retrieved by cosine similarity.

Retrieved steps are injected into agent prompts as history context so the
agent knows what has already been tried and what the outcomes were.

Model: all-MiniLM-L6-v2 (22M params, CPU-only, ~5ms per embed)
Loaded once at module import — not per session, to avoid startup latency
in the hot path of the session loop.
"""

import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

from core.session import StepRecord


# ── singleton model ───────────────────────────────────────────────────────────

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ── embedding helpers ─────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    """
    Embed a string and return a unit-normalized vector (shape: [384,]).
    Normalizing at store time means retrieval is just a dot product.
    """
    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _step_text(step: StepRecord, axtree_snippet: str) -> str:
    """
    Serialise a completed step into a short string for embedding.

    Format:
        action: <type> (<source>) | success: <true/false> | page: <axtree>

    The axtree is truncated to keep embedding text short — the first 400
    characters capture the most prominent elements on the page.
    """
    outcome = "true" if step.success else "false"
    snippet = axtree_snippet[:400].replace("\n", " ")
    return (
        f"action: {step.action_type} ({step.action_source}) | "
        f"success: {outcome} | "
        f"page: {snippet}"
    )


def _query_text(task: str, axtree_snippet: str) -> str:
    """
    Serialise the current page state + task into a query string for retrieval.
    """
    snippet = axtree_snippet[:400].replace("\n", " ")
    return f"task: {task} | page: {snippet}"


# ── memory store ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    step: StepRecord
    axtree_snippet: str     # raw text used when this step was stored
    vector: np.ndarray      # unit-normalized embedding


@dataclass
class EpisodicMemory:
    """
    In-memory episodic store for one session.

    One EpisodicMemory instance per Session — created at session start,
    discarded when the session ends. No persistence across process restarts.
    """
    entries: list[MemoryEntry] = field(default_factory=list)

    def store(self, step: StepRecord, axtree_snippet: str) -> None:
        """Embed and store a completed step."""
        text = _step_text(step, axtree_snippet)
        vec = _embed(text)
        self.entries.append(MemoryEntry(step=step, axtree_snippet=axtree_snippet, vector=vec))

    def retrieve(self, task: str, axtree_snippet: str, top_k: int = 3) -> list[MemoryEntry]:
        """
        Return the top-k most relevant past steps for the current page state.

        Similarity is cosine (dot product of unit vectors).
        Returns fewer than top_k if fewer steps have been stored.
        Returns an empty list at the first step (nothing stored yet).
        """
        if not self.entries:
            return []

        query_vec = _embed(_query_text(task, axtree_snippet))

        # Stack stored vectors into a matrix and compute similarities in one shot
        matrix = np.stack([e.vector for e in self.entries])  # shape: [n, 384]
        similarities = matrix @ query_vec                     # shape: [n,]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.entries[i] for i in top_indices]

    def format_for_prompt(self, entries: list[MemoryEntry]) -> str:
        """
        Format retrieved memory entries as a prompt-injectable history block.

        Example output injected into agent prompts:
            [Memory: relevant past steps]
            Step 1 — click (dom) → FAILED | page was: [1] button "Submit" ...
            Step 3 — type (dom) → OK    | page was: [2] textbox "Email" ...
        """
        if not entries:
            return ""

        lines = ["[Memory: relevant past steps]"]
        for entry in entries:
            outcome = "OK" if entry.step.success else "FAILED"
            snippet = entry.axtree_snippet[:200].replace("\n", " ")
            lines.append(
                f"Step {entry.step.step_index + 1} — "
                f"{entry.step.action_type} ({entry.step.action_source}) → {outcome} "
                f"| page was: {snippet}"
            )
        return "\n".join(lines)
