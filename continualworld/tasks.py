"""Canonical Continual World sequences using Meta-World v3 names."""

CW10: tuple[str, ...] = (
    "hammer-v3",
    "push-wall-v3",
    "faucet-close-v3",
    "push-back-v3",
    "stick-pull-v3",
    "handle-press-side-v3",
    "push-v3",
    "shelf-place-v3",
    "window-close-v3",
    "peg-unplug-side-v3",
)

# CW20 intentionally revisits the same ten tasks in the same order. Each
# occurrence still gets a distinct one-hot sequence identifier.
CW20: tuple[str, ...] = CW10 + CW10

TASK_SEQUENCES: dict[str, tuple[str, ...]] = {
    "CW10": CW10,
    "CW20": CW20,
}

# Name used by the original Continual World package.
TASK_SEQS = TASK_SEQUENCES


def resolve_sequence(sequence: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Resolve a named sequence or validate an explicit list of v3 task names."""
    if isinstance(sequence, str):
        try:
            return TASK_SEQUENCES[sequence.upper()]
        except KeyError as exc:
            choices = ", ".join(TASK_SEQUENCES)
            raise ValueError(
                f"Unknown sequence {sequence!r}; choose one of: {choices}"
            ) from exc

    tasks = tuple(sequence)
    if not tasks:
        raise ValueError("A continual-learning sequence must contain at least one task")
    return tasks
