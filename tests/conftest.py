"""Test-wide safety rails.

The one here that matters: **no test may write to the real custody ledger.**

`custody.record` is called from inside `ingest.pipeline.run`, `fusion.pipeline.run`
and the red-team path, which is correct — those are the moments a forensic
record has to capture. But it means every test that exercises the pipeline was
appending to `data/processed/custody.jsonl`, sealing files under
`/tmp/pytest-of-anish/...` that pytest then deleted. The result was a ledger of
522 entries, 396 of them pointing at temp files that no longer exist, and a
chain-of-custody page reporting hundreds of missing exhibits on a system where
nothing was wrong.

A forensic record full of test noise is not a forensic record. So the ledger is
redirected per test — and only when a test has not already chosen its own path,
so `test_custody.py` and `test_monitor.py` keep pointing at the files they
tamper with on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config
import custody


@pytest.fixture(scope="session", autouse=True)
def isolate_custody_ledger(tmp_path_factory):
    """Send anything aimed at the production ledger to one file for the session.

    Session-scoped on purpose. A function-scoped version caught most of it and
    still leaked eleven entries per run, because pytest builds module- and
    session-scoped fixtures *before* any function-scoped one — so every
    `tmp_path_factory` fixture that ingests a dataset had already written to the
    real ledger before the redirect was in place.
    """
    production = Path(config.load()["custody"]["ledger_path"]).resolve()
    original = custody.ledger_path
    sink = tmp_path_factory.mktemp("custody") / "custody.jsonl"

    def redirected(cfg: dict | None = None) -> Path:
        chosen = original(cfg)
        # A test that set its own path means it; only the real one is diverted.
        return sink if chosen.resolve() == production else chosen

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(custody, "ledger_path", redirected)
        yield sink


@pytest.fixture(scope="session", autouse=True)
def isolate_stacker_model(tmp_path_factory):
    """The same rail for the fitted stacker. `fusion.pipeline.run` saves it to
    `fusion.model_path` whenever its dataset has ground truth, so every test
    that runs the pipeline on a generated dataset was overwriting
    models/stacker.joblib — which the red-team path and `eval.report` then
    load. A suite run silently changed the canonical report's numbers."""
    from fusion import pipeline

    production = Path(config.load()["fusion"]["model_path"]).resolve()
    sink = tmp_path_factory.mktemp("models") / "stacker.joblib"
    original = pipeline.save

    def redirected(stacker, path):
        return original(stacker, sink if Path(path).resolve() == production else path)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(pipeline, "save", redirected)
        yield sink


#: Every production artifact the suite could reach. Hashed when the session
#: starts; tests/test_zz_artifacts.py checks them again at the end.
ARTIFACTS = [
    ("custody", "ledger_path"), ("fusion", "model_path"), ("fusion", "alerts_json"),
    ("fusion", "alerts_parquet"), ("fusion", "feedback_parquet"), ("fusion", "actors_json"),
    ("ingest", "output_path"), ("ingest", "quarantine_path"),
    ("features", "relay_path"), ("origination", "model_path"),
    ("engines", "correlation"),
]


def artifact_paths() -> list[Path]:
    cfg = config.load()
    paths = []
    for block, key in ARTIFACTS:
        value = cfg[block][key]
        paths.append(Path(value["output_path"] if isinstance(value, dict) else value))
    paths.append(Path(cfg["features"]["fingerprint"]["model_path"]))
    return paths


def artifact_hashes() -> dict[str, str | None]:
    import hashlib
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            for p in artifact_paths()}


@pytest.fixture(scope="session", autouse=True)
def artifacts_at_start():
    return artifact_hashes()

