"""
Session data structures for v3 multi-step task execution.

A Session represents one full run of POST /task — from initial page load
through termination. It holds the step history, checkpoints for rollback,
and the termination reason once the session ends.

This module is data-only for now. The session loop logic (observe → propose
→ rank → execute → check termination) will be added in Phase 3.
"""

from dataclasses import dataclass, field
from typing import Optional


# Valid termination reasons — every ended session must have one of these.
# "success" means the evaluator confirmed the task was completed.
# All others are failure modes and are the primary research artifact.
TERMINATION_REASONS = frozenset({
    "success",
    "action_failure",       # wrong element, coordinate drift, not interactable
    "cascade",              # one bad action corrupted subsequent page state
    "loop",                 # agent re-proposed same action, stuck
    "premature_termination",# evaluator said done when task was not complete
    "memory_miss",          # relevant past step not retrieved, mistake repeated
    "step_limit",           # soft step limit hit before any other signal
    "task_ambiguity",       # underspecified goal led to valid-but-wrong completion
})


@dataclass
class Checkpoint:
    """
    Snapshot of page state after a successful step.
    Used to restore the browser to a known-good state on rollback.
    """
    step_index: int
    url: str
    axtree_hash: str        # hash of the axtree at this step


@dataclass
class StepRecord:
    """
    Record of what happened at one step within a session.
    episode_id links back to the full candidates + execution data in the DB.
    """
    step_index: int
    episode_id: int             # FK → episodes.id
    page_url: str
    axtree_hash: str
    action_type: str            # e.g. "click", "type"
    action_source: str          # "dom" or "vision"
    success: bool
    action_text: Optional[str] = None    # text typed/selected, if any
    element_name: Optional[str] = None  # element label from axtree (dom agent only)


@dataclass
class Session:
    """
    Top-level object representing one multi-step task run.

    Created at the start of POST /task and updated after each step.
    Persisted to the `sessions` and `steps` tables via logger.py.
    """
    id: int                             # DB-assigned, set after insert
    task: str
    start_url: str
    steps: list[StepRecord] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    termination_reason: Optional[str] = None

    @property
    def current_step_index(self) -> int:
        return len(self.steps)

    @property
    def is_running(self) -> bool:
        return self.termination_reason is None

    def last_checkpoint(self) -> Optional[Checkpoint]:
        """Most recent checkpoint, or None if no successful steps yet."""
        return self.checkpoints[-1] if self.checkpoints else None

    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)

    def add_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.checkpoints.append(checkpoint)

    def terminate(self, reason: str) -> None:
        if reason not in TERMINATION_REASONS:
            raise ValueError(f"Unknown termination reason: {reason!r}")
        self.termination_reason = reason
