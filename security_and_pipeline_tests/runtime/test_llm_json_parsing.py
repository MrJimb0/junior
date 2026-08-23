"""_best_effort_json must survive the dominant local-model output shapes —
prose-wrapped JSON, ```json fences, truncated fences — and report a real error on
junk instead of silently returning (None, None).
"""
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.retrieve_and_prompt_step import (
    _best_effort_json,
)


def test_plain_json():
    data, err = _best_effort_json('{"a": 1, "b": "x"}')
    assert err is None
    assert data == {"a": 1, "b": "x"}


def test_prose_wrapped_fenced_json():
    # The classic small-Qwen shape: a preamble then a ```json fence.
    content = 'Sure! Here is the JSON:\n```json\n{"date_of_diagnosis": "2024-01-22"}\n```\nHope that helps.'
    data, err = _best_effort_json(content)
    assert err is None
    assert data == {"date_of_diagnosis": "2024-01-22"}


def test_bare_prose_then_object():
    data, err = _best_effort_json('The answer is {"second_opinion": false} based on the notes.')
    assert err is None
    assert data == {"second_opinion": False}


def test_truncated_closing_fence():
    # max_tokens cut the response before the closing ``` — still recoverable.
    data, err = _best_effort_json('```json\n{"genetic_testing_done": true, "genes": []}')
    assert err is None
    assert data == {"genetic_testing_done": True, "genes": []}


def test_junk_reports_error():
    data, err = _best_effort_json("I'm sorry, I cannot determine that from the chart.")
    assert data is None
    assert err  # a non-empty error string, not a silent (None, None)


def test_empty_reports_error():
    data, err = _best_effort_json("")
    assert data is None
    assert err
