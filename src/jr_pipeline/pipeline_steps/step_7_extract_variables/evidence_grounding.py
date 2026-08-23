"""One definition of what makes an evidence pointer trustworthy.

A value the pipeline extracts is only checkable if it points back at the passage it
came from. A model can satisfy that shape without satisfying the substance: it can
emit a well-formed chunk id for a passage it was never shown. A real run
did exactly that twice, citing ``clinical_note:368:0`` out of the
16 chunks in front of it.

A pointer is grounded when it names a chunk actually put in front of the model, or is
a structured evidence pointer the pipeline authored itself (``patient:...:structured:``,
which carries its own file+row hash and so verifies without a shown-chunk list).

This lives in its own module because two places need the same answer and must not
drift: the prompt step checks its own answer while the valid ids are still in hand and
can ask again, and ``extract`` re-checks the assembled result against every step's
shown chunks as the last word before anything is written.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    parse_structured_evidence_pointer,
)


# The key test step 8 uses, kept identical here on purpose: the exact key, a prefixed
# key (``dob_evidence_chunk_id``), or a per-item list (``evidence_chunk_ids``).
def is_evidence_pointer_key(key: str) -> bool:
    """True for any key that carries a pointer back to source text."""
    return "evidence_chunk_id" in key.lower()


def is_grounded(chunk_id: str, shown_chunk_ids: set[str]) -> bool:
    """True when this id names a passage the model was actually shown, or is a
    pipeline-authored structured pointer that verifies on its own."""
    return (
        chunk_id in shown_chunk_ids
        or parse_structured_evidence_pointer(chunk_id) is not None
    )


def find_ungrounded_pointers(
    data: Any, shown_chunk_ids: set[str], path: str = "data"
) -> list[dict[str, str]]:
    """Every evidence pointer in ``data`` citing something not in ``shown_chunk_ids``.

    Read-only: it reports, it does not null anything. The prompt step uses it to decide
    whether asking again is worth a call; ``extract`` does the nulling later, once no
    further attempt will be made."""
    found: list[dict[str, str]] = []

    def walk(node: Any, node_path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if is_evidence_pointer_key(key):
                    candidates = value if isinstance(value, list) else [value]
                    for item in candidates:
                        if isinstance(item, str) and item and not is_grounded(item, shown_chunk_ids):
                            found.append({"path": f"{node_path}.{key}", "chunk_id": item})
                elif isinstance(value, (dict, list)):
                    walk(value, f"{node_path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{node_path}[{index}]")

    walk(data, path)
    return found


def correction_prompt(ungrounded: list[dict[str, str]], shown_chunk_ids: list[str]) -> str:
    """What to say to a model that cited a passage it was never shown.

    It names the offending ids rather than describing the rule in the abstract, and it
    says plainly that returning null is an acceptable answer. Without that last line a
    model under a "cite something from this list" instruction will pick a plausible id
    to satisfy the constraint, which converts a caught failure into an uncaught one --
    the pointer then passes every automatic check and only a human reading the passage
    would find it does not support the value."""
    invented = sorted({entry["chunk_id"] for entry in ungrounded})
    return (
        "Your previous answer cited "
        + ", ".join(repr(chunk_id) for chunk_id in invented)
        + ", which "
        + ("is not one of" if len(invented) == 1 else "are not among")
        + " the passages you were given. Answer again using only these chunk ids:\n"
        + "\n".join(f"  {chunk_id}" for chunk_id in shown_chunk_ids)
        + "\n\nIf no passage above supports a value, return null for that value and null "
        "for its evidence_chunk_id. A null answer is correct when the passages do not "
        "say; a citation to a passage you were not given is not."
    )


def find_uncited_values(data: Any) -> list[str]:
    """Every object holding a value with no evidence pointer of its own.

    The same walk step 8 runs, imported rather than restated so a step re-asks for
    exactly what the gate would otherwise fail it for. A step is only asked again when
    its answer carries pointer KEYS at all: a gather or summarise step that was never
    meant to cite anything would otherwise be re-asked until the budget ran out and
    never satisfy a question nobody put to it."""
    from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
        find_unprovenanced_value_paths,
    )

    if not _mentions_a_pointer(data):
        return []
    return find_unprovenanced_value_paths(data)


def _mentions_a_pointer(node: Any) -> bool:
    if isinstance(node, dict):
        return any(is_evidence_pointer_key(k) or _mentions_a_pointer(v) for k, v in node.items())
    if isinstance(node, list):
        return any(_mentions_a_pointer(item) for item in node)
    return False


def correction_prompt_for_uncited(uncited: list[str], shown_chunk_ids: list[str]) -> str:
    """What to say to a model that gave a value and cited nothing for it.

    Says the same last thing the ungrounded correction says, and for the same reason: a
    model pressed to produce a citation will produce a plausible one. A null value is a
    better answer than a value with a borrowed source, because the first is visibly
    unanswered and the second is invisibly wrong."""
    return (
        "Your previous answer gave a value at "
        + ", ".join(sorted(uncited))
        + " with no evidence_chunk_id for it. Every value must name the passage it came "
        "from. Answer again using only these chunk ids:\n"
        + "\n".join(f"  {chunk_id}" for chunk_id in shown_chunk_ids)
        + "\n\nIf no passage above supports a value, return null for that value and null "
        "for its evidence_chunk_id. A null answer is correct when the passages do not "
        "say; a value with no source, or one borrowed from an unrelated passage, is not."
    )


def ask_until_grounded(
    *,
    call_once: Any,
    messages: list[dict[str, str]],
    shown_chunk_ids: set[str],
    max_retries: int,
    parse: Any,
) -> dict[str, Any]:
    """Ask, and re-ask while the answer cites passages that were never shown, or gives
    a value it cites nothing for.

    Every prompt step shares this. They differ only in what counts as shown -- a
    retrieval step shows its evidence packet, a per-row step shows the one row, a step
    with no retrieval of its own shows whatever the steps before it established -- so
    that set is the caller's to supply, and everything after it is the same everywhere.
    Three copies of this loop is how the three would drift apart.

    ``call_once(messages) -> (response, was_cached)`` keeps each step's own cache
    handling where it already lives. ``parse(content) -> (parsed, parse_error)`` likewise.

    Returns the final response with the attempts that preceded it. A model that keeps
    inventing is not argued with forever: after the budget the last answer stands, its
    ungrounded pointer is nulled downstream, and the value fails as an unsupported claim.
    A value still uncited after the budget is left uncited, for the same reason — the
    gate reports it, and a reader sees an unanswered value rather than a value wearing
    somebody else's source.
    """
    conversation = list(messages)
    attempts: list[dict[str, Any]] = []
    while True:
        response, was_cached = call_once(conversation)
        parsed, parse_error = parse(response.content)

        if parse_error is not None or len(attempts) >= max(0, int(max_retries or 0)):
            break
        # Two ways an answer fails to be checkable, and they are asked about
        # differently. A citation to a passage never shown is the louder one and is
        # corrected first; a value with no citation at all is the quieter one, and used
        # to be left for a recipe helper to paper over by attaching whatever passage was
        # nearest. Both are the same failure to a reader: a value that cannot be
        # checked.
        ungrounded = find_ungrounded_pointers(parsed, shown_chunk_ids)
        uncited = [] if ungrounded else find_uncited_values(parsed)
        if not ungrounded and not uncited:
            break
        attempts.append({
            "attempt": len(attempts) + 1,
            "ungrounded": ungrounded,
            "uncited": uncited,
            "content": response.content,
        })
        correction = (
            correction_prompt(ungrounded, sorted(shown_chunk_ids)) if ungrounded
            else correction_prompt_for_uncited(uncited, sorted(shown_chunk_ids))
        )
        conversation = conversation + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": correction},
        ]
    return {
        "response": response,
        "parsed": parsed,
        "parse_error": parse_error,
        "attempts": attempts,
        "was_cached": was_cached,
    }
