from core.session import Checkpoint, Session, StepRecord
from core.rollback import (
    CASCADE_THRESHOLD,
    consecutive_failures,
    should_cascade,
    _find_checkpoint,
)


def make_session() -> Session:
    return Session(id=1, task="test task", start_url="https://example.com")


def make_step(index: int, success: bool) -> StepRecord:
    return StepRecord(
        step_index=index,
        episode_id=index + 1,
        page_url="https://example.com",
        axtree_hash=f"hash{index}",
        action_type="click",
        action_source="dom",
        success=success,
        action_text=None,
        element_name=None,
    )


def make_checkpoint(step_index: int) -> Checkpoint:
    return Checkpoint(
        step_index=step_index,
        url="https://example.com",
        axtree_hash=f"hash{step_index}",
    )


class TestConsecutiveFailures:
    def test_no_steps(self):
        session = make_session()
        assert consecutive_failures(session) == 0

    def test_all_success(self):
        session = make_session()
        for i in range(3):
            session.add_step(make_step(i, success=True))
        assert consecutive_failures(session) == 0

    def test_all_failures(self):
        session = make_session()
        for i in range(4):
            session.add_step(make_step(i, success=False))
        assert consecutive_failures(session) == 4

    def test_mixed_ends_in_failures(self):
        session = make_session()
        session.add_step(make_step(0, success=True))
        session.add_step(make_step(1, success=False))
        session.add_step(make_step(2, success=False))
        assert consecutive_failures(session) == 2

    def test_mixed_ends_in_success(self):
        session = make_session()
        session.add_step(make_step(0, success=False))
        session.add_step(make_step(1, success=False))
        session.add_step(make_step(2, success=True))
        assert consecutive_failures(session) == 0


class TestShouldCascade:
    def test_no_checkpoint_never_cascades(self):
        session = make_session()
        for i in range(CASCADE_THRESHOLD + 1):
            session.add_step(make_step(i, success=False))
        # No checkpoint → nothing to roll back to → should not cascade
        assert should_cascade(session) is False

    def test_below_threshold(self):
        session = make_session()
        session.add_checkpoint(make_checkpoint(0))
        for i in range(CASCADE_THRESHOLD - 1):
            session.add_step(make_step(i, success=False))
        assert should_cascade(session) is False

    def test_at_threshold(self):
        session = make_session()
        session.add_checkpoint(make_checkpoint(0))
        for i in range(CASCADE_THRESHOLD):
            session.add_step(make_step(i, success=False))
        assert should_cascade(session) is True

    def test_above_threshold(self):
        session = make_session()
        session.add_checkpoint(make_checkpoint(0))
        for i in range(CASCADE_THRESHOLD + 2):
            session.add_step(make_step(i, success=False))
        assert should_cascade(session) is True


class TestFindCheckpoint:
    def test_no_checkpoints(self):
        session = make_session()
        assert _find_checkpoint(session, to_step=5) is None

    def test_exact_match(self):
        session = make_session()
        session.add_checkpoint(make_checkpoint(2))
        result = _find_checkpoint(session, to_step=2)
        assert result is not None
        assert result.step_index == 2

    def test_returns_latest_at_or_before(self):
        session = make_session()
        session.add_checkpoint(make_checkpoint(1))
        session.add_checkpoint(make_checkpoint(3))
        session.add_checkpoint(make_checkpoint(5))
        result = _find_checkpoint(session, to_step=4)
        assert result is not None
        assert result.step_index == 3

    def test_all_checkpoints_after_step(self):
        session = make_session()
        session.add_checkpoint(make_checkpoint(5))
        assert _find_checkpoint(session, to_step=3) is None
