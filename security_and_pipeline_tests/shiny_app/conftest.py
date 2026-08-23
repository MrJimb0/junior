"""Put the workbench app folder on sys.path for these tests, and boot the app.

The app ships as a runnable folder (``apps_and_interfaces/shiny_review_app``),
not an installed package: its modules import each other flat (``import
run_results``, ``import run_inspection``). So the only way to import them is to
add the folder itself to ``sys.path`` — which is exactly what ``shiny run`` does
implicitly. The suite-wide ``JR_*`` env snapshot/restore lives in the parent
conftest.

``review_screen`` runs the app's own ``server()`` so the button handlers and the
rendered panels can be exercised. Everything a reviewer can do irreversibly — every
feedback write — lives inside those handlers as a closure, so a test that only calls
the module-level helpers can show that a guard returns the right sentence but never
that the click it guards wrote nothing.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2] / "apps_and_interfaces" / "shiny_review_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# The workbench is the `app` extra, which the dev install does not pull in. Skipping the
# folder here is what keeps one absent dependency from failing collection for the whole
# suite: a collection error reports no tests at all, so the 1400 that would have passed
# read as nothing having run.
pytest.importorskip("shiny", reason="needs the app extra: pip install -e '.[app]'")

import app as review_app  # noqa: E402 — only importable once the folder is on sys.path
import run_results  # noqa: E402
from shiny import reactive  # noqa: E402
from shiny.session import Inputs, session_context  # noqa: E402

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (  # noqa: E402
    extract_output_dir,
    phi_patient_run_dir,
)


class ReviewScreen:
    """One booted review app, driven the way a browser drives it.

    The session and the output registry are stand-ins: nothing is sent to a client, so
    a click is an input value changing and a flush, and a panel is its render function
    called by name. What runs in between is the app's own code.
    """

    def __init__(self) -> None:
        self._session = MagicMock()
        self._session.ns = ""  # the root namespace; the app declares no modules
        self._inputs = Inputs({})
        self._click_counts: dict[str, int] = {}
        self._panels: dict[str, object] = {}
        registered_outputs = MagicMock()
        with session_context(self._session):
            review_app.server(self._inputs, registered_outputs, self._session)
            asyncio.run(reactive.flush())
        for call in registered_outputs.call_args_list:
            self._panels[call.args[0].__name__] = call.args[0]

    def set_input(self, input_id: str, value: object) -> None:
        """Type into a text box or choose from a select."""
        with session_context(self._session):
            self._inputs[input_id]._set(value)
            asyncio.run(reactive.flush())

    def click_button(self, button_id: str) -> None:
        """Press a button: an action button's value is its click count."""
        self._click_counts[button_id] = self._click_counts.get(button_id, 0) + 1
        self.set_input(button_id, self._click_counts[button_id])

    def panel_text(self, output_id: str) -> str:
        """What one panel currently renders, as the text a reviewer would read."""
        async def _render():
            with session_context(self._session), reactive.isolate():
                return await self._panels[output_id].fn()

        return _readable_text(asyncio.run(_render()))


def _readable_text(node: object) -> str:
    """Flatten a rendered panel to the words on screen. ``ui.HTML`` is already a
    string; ``ui.tags.*`` nest their children."""
    if node is None:
        return ""
    if isinstance(node, str):
        return str(node)
    children = getattr(node, "children", None)
    if children is None:
        return str(node)
    return "".join(_readable_text(child) for child in children)


@pytest.fixture
def review_screen(tmp_path, monkeypatch):
    """Boot the review app against a temporary data root, opened on a given run+patient.

    The run and patient stand in for what the sidebar selects were seeded with at boot,
    which is how the app reaches every state a reviewer can be in: ``(None, None)`` is
    nothing selected, a run with no patient is a run whose patient folders are gone,
    and a run+patient with extract output on disk is a real run.
    """
    def boot(run_id: str | None = None, patient_id: str | None = None) -> ReviewScreen:
        monkeypatch.setattr(run_results, "DATA_ROOT", tmp_path)
        monkeypatch.setattr(review_app, "ACTIVE_DR", tmp_path)
        monkeypatch.setattr(review_app, "DEFAULT_RUN_ID", run_id)
        monkeypatch.setattr(review_app, "DEFAULT_PATIENT_ID", patient_id)
        return ReviewScreen()

    return boot


@pytest.fixture
def write_extract_output(tmp_path):
    """Fabricate ``<receipts>/<run>/patients/<pid>/extract/<var>/result.json`` under the
    temporary data root, through the same path helpers the loader reads with."""
    def write(run_id: str, patient_id: str, variable: str,
              value: str = "1970-01-01", evidence_chunk_id: str | None = None) -> None:
        data: dict[str, object] = {variable: {"type": "string", "value": value}}
        if evidence_chunk_id:
            data[f"{variable}_evidence_chunk_id"] = evidence_chunk_id
        envelope = {"payload": {
            "variable": variable, "ok": True, "data": data,
            "recipe": {"name": "demographics", "version": "1"},
        }}
        var_dir = extract_output_dir(phi_patient_run_dir(run_id, patient_id, tmp_path)) / variable
        var_dir.mkdir(parents=True, exist_ok=True)
        (var_dir / "result.json").write_text(json.dumps(envelope), encoding="utf-8")

    return write
