"""Every API endpoint: alerts, entity detail, subgraph, PDF, feedback, stats.

Two fixtures. `client` runs the real pipeline over a small generated dataset,
because entity detail and the subgraph need a real graph behind them.
`fixture_client` points the app at tests/fixtures/final_alerts.json, so the
listing, filtering, pagination and feedback endpoints are tested against known
rows instead of whatever the generator happened to produce.
"""

from __future__ import annotations

import json

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import config

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient            # noqa: E402

import api.app as app_module                        # noqa: E402

CFG = config.load()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fusion.pipeline import run as fusion_run
    from generator.main import build_parser, generate
    from ingest.pipeline import run as ingest_run

    d = tmp_path_factory.mktemp("api")
    raw = d / "raw"
    generate(build_parser().parse_args(["--n-actors", "120", "--n-transactions", "600",
                                        "--output", str(raw), "--seed", "23",
                                        "--formats", "csv"]))
    ingest_run(raw, d / "t.parquet", d / "q.parquet", "csv")
    fusion_run(d / "t.parquet", raw / "ground_truth.json", d / "final.parquet",
               d / "final.json", CFG)
    app_module.configure(d / "t.parquet", raw, d / "final.json", d / "feedback.parquet")
    yield TestClient(app_module.app), d
    app_module.configure()


FIXTURE = Path(__file__).parent / "fixtures" / "final_alerts.json"


@pytest.fixture
def fixture_client(client, tmp_path):
    """The same app, reading the hand-written alert fixture."""
    _, shared = client
    app_module.configure(shared / "t.parquet", shared / "raw", FIXTURE,
                         tmp_path / "feedback.parquet")
    try:
        yield TestClient(app_module.app), tmp_path / "feedback.parquet"
    finally:
        app_module.configure(shared / "t.parquet", shared / "raw",
                             shared / "final.json", shared / "feedback.parquet")


def an_alerted_entity(api) -> str:
    return api.get("/alerts?limit=1").json()["alerts"][0]["entity_id"]


def a_multi_hop_txid(directory) -> str:
    df = pd.read_parquet(directory / "t.parquet")
    counts = df.groupby("txid").size()
    return str(counts[counts > 2].index[0])


def test_propagation_returns_cytoscape_elements(client):
    api, d = client
    txid = a_multi_hop_txid(d)
    body = api.get(f"/transactions/{txid}/propagation").json()
    assert body["txid"] == txid
    assert set(body) >= {"estimated_origin", "ip_class", "confidence", "runner_ups",
                         "elements", "layout", "caveat"}
    nodes, edges = body["elements"]["nodes"], body["elements"]["edges"]
    assert nodes and edges
    for node in nodes:                                  # Cytoscape shape
        assert set(node["data"]) >= {"id", "label", "role", "ip_class", "badge"}
    ids = {n["data"]["id"] for n in nodes}
    for edge in edges:
        assert edge["data"]["source"] in ids and edge["data"]["target"] in ids


def test_origin_and_runner_ups_are_marked_for_the_dashboard(client):
    api, d = client
    body = api.get(f"/transactions/{a_multi_hop_txid(d)}/propagation").json()
    roles = {n["data"]["id"]: n["data"]["role"] for n in body["elements"]["nodes"]}
    assert roles[body["estimated_origin"]] == "origin"
    assert list(roles.values()).count("origin") == 1
    for runner in body["runner_ups"]:
        assert roles[runner["ip"]] == "runner_up"


def test_tree_is_rooted_at_the_estimated_origin_for_dagre(client):
    api, d = client
    body = api.get(f"/transactions/{a_multi_hop_txid(d)}/propagation").json()
    assert body["layout"] == {"name": "dagre", "roots": [body["estimated_origin"]]}


def test_every_node_carries_an_ip_class_badge(client):
    api, d = client
    body = api.get(f"/transactions/{a_multi_hop_txid(d)}/propagation").json()
    allowed = {"relay", "tor", "hosting", "residential"}
    assert {n["data"]["badge"] for n in body["elements"]["nodes"]} <= allowed
    assert all(n["data"]["evidence"] for n in body["elements"]["nodes"])


def test_response_states_the_attribution_caveat(client):
    api, d = client
    body = api.get(f"/transactions/{a_multi_hop_txid(d)}/propagation").json()
    assert "not an attribution" in body["caveat"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_unknown_transaction_is_a_404(client):
    api, _ = client
    assert api.get("/transactions/deadbeef/propagation").status_code == 404


def test_stats_reports_origin_estimation_status(client):
    api, _ = client
    body = api.get("/stats").json()
    origin = body["features"]["origin_estimation"]
    assert origin["status"] == "ok"
    assert origin["multi_hop_transactions"] > 0
    assert origin["estimator"] == CFG["engines"]["propagation"]["estimator"]
    assert body["alerts"] > 0


def test_stats_flags_degraded_mode_on_single_row_data(client, tmp_path):
    """The dashboard must be able to say origin estimation is not really running."""
    from generator.main import build_parser, generate
    from ingest.pipeline import run as ingest_run

    _, shared = client                    # restore this afterwards, not the defaults
    raw = tmp_path / "raw"
    generate(build_parser().parse_args(["--n-actors", "40", "--n-transactions", "150",
                                        "--output", str(raw), "--seed", "5",
                                        "--formats", "csv", "--single-row"]))
    ingest_run(raw, tmp_path / "t.parquet", tmp_path / "q.parquet", "csv")
    app_module.configure(tmp_path / "t.parquet", raw, tmp_path / "missing.json")
    try:
        body = TestClient(app_module.app).get("/stats").json()
        origin = body["features"]["origin_estimation"]
        assert origin["status"] == "degraded"
        assert "single relay record" in origin["reason"]
        assert origin["multi_hop_transactions"] == 0
    finally:
        app_module.configure(shared / "t.parquet", shared / "raw", shared / "final.json")


def test_alerts_endpoint_serves_the_fusion_output(client):
    api, _ = client
    body = api.get("/alerts?limit=5").json()
    assert len(body["alerts"]) <= 5
    assert body["alerts"] and body["alerts"][0]["reason"].startswith("Flagged due to")
    assert "warning" in body["stacker"]


# --- /alerts --------------------------------------------------------------
def test_alerts_are_ranked_by_risk_score(fixture_client):
    api, _ = fixture_client
    body = api.get("/alerts").json()
    scores = [a["risk_score"] for a in body["alerts"]]
    assert scores == sorted(scores, reverse=True)
    assert body["total"] == 3


def test_alerts_paginate(fixture_client):
    api, _ = fixture_client
    first = api.get("/alerts?limit=2&offset=0").json()
    second = api.get("/alerts?limit=2&offset=2").json()
    assert len(first["alerts"]) == 2 and len(second["alerts"]) == 1
    assert first["total"] == second["total"] == 3
    ids = [a["alert_id"] for a in first["alerts"] + second["alerts"]]
    assert len(set(ids)) == 3


def test_alerts_filter_by_score_type_and_pattern(fixture_client):
    api, _ = fixture_client
    assert {a["alert_id"] for a in api.get("/alerts?min_score=0.6").json()["alerts"]} == \
        {"C000001", "C000002"}
    assert {a["alert_id"] for a in api.get("/alerts?entity_type=wallet").json()["alerts"]} == \
        {"bc1qsolo"}
    by_pattern = api.get("/alerts?pattern_type=layering").json()
    assert [a["alert_id"] for a in by_pattern["alerts"]] == ["C000002"]
    assert by_pattern["filters"]["pattern_type"] == "layering"


# --- /entities ------------------------------------------------------------
def test_entity_detail_carries_features_scores_and_explanation(client):
    api, _ = client
    entity_id = an_alerted_entity(api)
    body = api.get(f"/entities/{entity_id}").json()
    assert body["entity_id"] == entity_id and body["alerted"]
    assert body["features"] and "txs" in body["features"]
    assert set(body["scores"]) >= {"risk_score", "rule_score", "anomaly_score",
                                   "gnn_score", "taint_score", "contributions"}
    assert body["reason"] and body["wallets"]
    assert isinstance(body["taint_path"], list) and isinstance(body["leads"], list)
    assert body["entity_type"] in ("cluster", "wallet")
    assert "not cleared" in body["caveat"]


def test_an_unknown_entity_is_a_404(client):
    api, _ = client
    assert api.get("/entities/nope").status_code == 404
    assert api.get("/entities/nope/graph").status_code == 404


def test_subgraph_is_cytoscape_ready_and_hop_bounded(client):
    api, _ = client
    entity_id = an_alerted_entity(api)
    small = api.get(f"/entities/{entity_id}/graph?hops=1").json()
    big = api.get(f"/entities/{entity_id}/graph?hops=3").json()

    assert {n["data"]["type"] for n in small["elements"]["nodes"]} <= {"wallet",
                                                                      "transaction", "ip"}
    for node in small["elements"]["nodes"]:
        assert {"id", "label", "type", "hop"} <= set(node["data"])
        assert node["data"]["hop"] <= 1
    ids = {n["data"]["id"] for n in small["elements"]["nodes"]}
    for edge in small["elements"]["edges"]:
        assert edge["data"]["source"] in ids and edge["data"]["target"] in ids
        assert edge["data"]["type"]
    assert len(big["elements"]["nodes"]) >= len(small["elements"]["nodes"])
    assert any(n["data"]["is_focus"] for n in small["elements"]["nodes"])


# --- /report --------------------------------------------------------------
def test_report_returns_a_one_page_pdf(client):
    api, _ = client
    response = api.get(f"/entities/{an_alerted_entity(api)}/report")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-") and response.content.endswith(b"%%EOF\n")
    assert b"/Count 1" in response.content            # exactly one page
    assert len(response.content) > 1500               # it drew something


# --- /feedback ------------------------------------------------------------
def test_feedback_appends_a_row_per_verdict(fixture_client):
    api, path = fixture_client
    assert api.post("/alerts/C000001/feedback", json={"status": "confirmed"}).status_code == 200
    api.post("/alerts/C000002/feedback", json={"status": "false_positive"})
    api.post("/alerts/C000001/feedback", json={"status": "false_positive"})  # changed mind

    rows = pd.read_parquet(path)
    assert len(rows) == 3                       # appended, never updated
    assert list(rows["status"]) == ["confirmed", "false_positive", "false_positive"]
    assert set(rows["alert_id"]) == {"C000001", "C000002"}
    assert rows["recorded_at"].is_monotonic_increasing


def test_feedback_rejects_an_unknown_alert_and_a_bad_status(fixture_client):
    api, _ = fixture_client
    assert api.post("/alerts/nope/feedback", json={"status": "confirmed"}).status_code == 404
    assert api.post("/alerts/C000001/feedback", json={"status": "maybe"}).status_code == 422


# --- /stats ---------------------------------------------------------------
def test_stats_summarises_the_dashboard_header(fixture_client):
    api, _ = fixture_client
    body = api.get("/stats").json()
    assert body["total_alerts"] == 3
    assert body["total_entities"] > 0
    assert body["alerts_by_pattern_type"]["layering"] == 1
    assert body["alerts_by_pattern_type"]["ransomware_collector"] == 1
    assert body["avg_confidence"] == pytest.approx((0.91 + 0.64 + 0.52) / 3, abs=1e-3)


def test_version_reports_the_build_and_the_route_count(client):
    """The console compares this with the commit compiled into its bundle."""
    api, _ = client
    body = api.get("/version").json()
    assert set(body) == {"commit", "started_at", "routes"}
    assert body["commit"] and len(body["commit"]) <= 40
    assert body["routes"] >= 10, "routes are counted from the schema, so routers count too"
    # Parsable, and in the past.
    started = datetime.fromisoformat(body["started_at"])
    assert started <= datetime.now(timezone.utc)


def test_version_prefers_the_stamped_commit(monkeypatch, client):
    """A container built without .git passes BTC_INTEL_COMMIT instead."""
    api, _ = client
    monkeypatch.setenv("BTC_INTEL_COMMIT", "deadbeefcafe")
    app_module._commit.cache_clear()
    try:
        assert api.get("/version").json()["commit"] == "deadbee"
    finally:
        app_module._commit.cache_clear()


# --- no origin leaves the API without a validity verdict ----------------------
ORIGIN_KEYS = ("estimated_origin", "estimated_origin_ip")


def origins_in(payload, found=None) -> list[dict]:
    """Every dict in a response that names an origin: an estimate, or a lead."""
    found = [] if found is None else found
    if isinstance(payload, dict):
        if any(k in payload for k in ORIGIN_KEYS) or ("ip" in payload and "observations" in payload):
            found.append(payload)
        for key, value in payload.items():
            if key == "leads" and isinstance(value, str):
                value = json.loads(value)
            origins_in(value, found)
    elif isinstance(payload, list):
        for item in payload:
            origins_in(item, found)
    return found


def assert_verdict(origin: dict) -> None:
    verdict = origin.get("validity")
    assert isinstance(verdict, dict), f"origin without a validity verdict: {origin}"
    assert verdict["tier"] in ("PASS", "ABSTAIN", "QUALIFIED", "ANNOTATE", "NOT_ASSESSED")
    if verdict["tier"] == "PASS":
        assert verdict["reason"] is None
    else:
        assert verdict["reason"] and verdict["evidence"], origin


def test_no_origin_leaves_the_api_without_a_validity_verdict(client):
    api, d = client
    df = pd.read_parquet(d / "t.parquet")
    seen = []
    for txid in df["txid"].drop_duplicates().head(40):
        body = api.get(f"/transactions/{txid}/propagation").json()
        assert "probability" in body and body["calibration_basis"]
        if body["validity"]["tier"] == "ABSTAIN":
            assert body["low_confidence_origin"], "a withheld origin must abstain"
            assert body["answer"] is None
        seen += origins_in(body)
    listing = api.get("/alerts?limit=50").json()
    seen += origins_in(listing)
    for alert in listing["alerts"][:10]:
        seen += origins_in(api.get(f"/entities/{alert['entity_id']}").json())
    assert any("observations" in o for o in seen), "no lead was exercised"
    assert len(seen) >= 40
    for origin in seen:
        assert_verdict(origin)


def test_a_lead_stored_before_the_validity_layer_is_never_served_as_pass(fixture_client):
    api, _ = fixture_client
    leads = origins_in(api.get("/alerts").json())
    assert leads
    for lead in leads:
        assert_verdict(lead)
        assert lead["validity"]["tier"] == "NOT_ASSESSED"


def test_no_coinjoin_answer_attributes_input_ownership(client):
    api, d = client
    truth = json.loads((d / "raw" / "ground_truth.json").read_text())["transactions"]
    mixes = [t for t, meta in truth.items() if meta["pattern"] == "coinjoin"]
    df = pd.read_parquet(d / "t.parquet")
    mixes = [t for t in mixes if (df["txid"] == t).sum() > 1][:10]
    assert mixes, "the generated dataset has no multi-hop CoinJoin"
    for txid in mixes:
        body = api.get(f"/transactions/{txid}/propagation").json()
        assert "COINJOIN" in body["validity"]["reasons"]
        answer = body["answer"]
        if answer is not None:                  # withheld is fine too
            assert answer["kind"] != "ip_attribution"
            assert answer["input_ownership"].startswith("not attributable")


ONION = "pg6mmjiyjmcrsslvykfwnntlaru7p5svn6y2ymmju6nubxndf4pscryd.onion"


def test_no_tor_onion_answer_exposes_an_ip(client, tmp_path):
    import ipaddress
    api, d = client
    df = pd.read_parquet(d / "t.parquet")
    counts = df.groupby("txid").size()
    source = df[df["txid"] == counts[counts > 3].index[0]].sort_values("timestamp").copy()
    first = source["src_ip"].iloc[0]
    source["txid"] = "0" * 63 + "1"
    source["src_ip"] = source["src_ip"].replace(first, ONION)
    source["dst_ip"] = source["dst_ip"].replace(first, ONION)
    pd.concat([df, source], ignore_index=True).to_parquet(tmp_path / "t.parquet")
    app_module.configure(tmp_path / "t.parquet", d / "raw", d / "final.json",
                         tmp_path / "feedback.parquet")
    try:
        body = TestClient(app_module.app).get(f"/transactions/{'0' * 63 + '1'}/propagation").json()
    finally:
        app_module.configure(d / "t.parquet", d / "raw", d / "final.json", d / "feedback.parquet")
    assert body["estimated_origin"] == ONION
    assert "TOR_ONION" in body["validity"]["reasons"]
    answer = body["answer"]
    assert answer["kind"] == "onion_identity" and "ip" not in answer
    for value in answer.values():
        try:
            ipaddress.ip_address(str(value))
        except ValueError:
            continue
        raise AssertionError(f"an onion answer carries an IP: {answer}")


# --- actors ----------------------------------------------------------------
def test_actors_are_served_and_every_view_and_verdict_is_in_the_ledger(client):
    import custody
    c, d = client
    body = c.get("/actors", params={"alerted_only": "false"}).json()
    assert body["total"] >= 1 and "does not name or imply a person" in body["statement"]
    first = body["actors"][0]
    assert first["name"] == f"actor {first['actor_id']}"
    detail = c.get(f"/actors/{first['actor_id']}").json()
    assert set(detail["drill_down"]["entities"]) == set(first["members"])
    assert all(v.startswith("/entities/") for v in detail["drill_down"]["entities"].values())
    views = [e for e in custody.read() if e["action"] == "actor.view"]
    assert views[-1]["detail"]["subject"] == first["actor_id"] and views[-1]["detail"]["found"]
    assert c.get("/actors/A-999999").status_code == 404
    assert custody.read()[-1]["detail"] == {**custody.read()[-1]["detail"], "found": False}
    verdict = c.post(f"/actors/{first['actor_id']}/verdict", json={"status": "confirmed"}).json()
    assert verdict["recorded"] == 1
    last = custody.read()[-1]
    assert last["action"] == "actor.verdict" and last["detail"]["actor_id"] == first["actor_id"]
    assert (d / "actor_feedback.parquet").exists() and not (d / "feedback.parquet").exists()
