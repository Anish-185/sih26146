"""Local read-only API over the processed artefacts.

    uvicorn api.app:app --host 127.0.0.1 --port 8000

Serves what the pipeline already wrote to data/processed/. No analysis happens
here beyond shaping subgraphs and a propagation tree for the dashboard, and
nothing reaches the network — the whole system stays air-gapped.

NO AUTHENTICATION. Deliberate, for the hackathon demo: this binds to localhost
and serves synthetic data. A real deployment carries case data for live
investigations and would need, at minimum, authentication with per-case
authorisation on every endpoint below, an audit log of who read which entity
(which is itself evidence), TLS termination, and a CORS list that is not a
convenience for a dev server. The feedback endpoint would additionally need the
analyst's identity recorded with the verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

import config
import custody
from analysis import validity
from analysis.validity import NOT_ASSESSED
from engines.propagation.estimators import CALIBRATION_BASIS, estimate_origin
from graph.builder import iter_transactions
from engines.propagation.tree import build_trees, degraded_mode
from engines.correlation import profile as correlation_profile
from features import fingerprint
from engines.rules.detectors import FeatureSet
from origination.model import OriginationModel
from graph.builder import IP, TRANSACTION, WALLET, build_graph, load
from ingest.ip_intel import load_intel

from fusion import incremental
from fusion.ordering import NO_TAINT, TIEBREAKERS
from fusion.pipeline import collect_signals

from . import graph as graph_api
from . import monitor as monitor_api
from . import redteam as redteam_api
from .case_report import render_pdf

app = FastAPI(title="btc-intel", version="0.1.0",
              description="Offline Bitcoin transaction forensics (SIH26146)")

# Local frontend dev. See the module docstring: not what a deployment does.
app.add_middleware(CORSMiddleware, allow_origins=config.get("api.cors_origins"),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

CLASS_BADGE = {"known_bitcoin_relay": "relay", "tor_exit": "tor",
               "hosting_vpn": "hosting", "residential_or_unknown": "residential"}

STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")


@lru_cache(maxsize=1)
def _commit() -> str:
    """Which build this process is. Read once, at start.

    A demo is lost when the browser talks to a server started before the fix
    it is demonstrating — the page looks right and the data is old. The commit
    is stamped into both halves so they can be compared instead of trusted.
    BTC_INTEL_COMMIT wins, for a container built without a .git directory.
    """
    stamped = os.environ.get("BTC_INTEL_COMMIT")
    if stamped:
        return stamped.strip()[:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                             cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=2, check=True)
        return out.stdout.strip()
    except Exception:                      # no git, no repo, no problem
        return "unknown"

# Where the API reads from. Defaults come from config.yaml; configure() points
# the app at another set of artefacts (a second case, or a test fixture).
STATE: dict = {"transactions": None, "intel_dir": None, "alerts_json": None,
               "feedback": None, "relay": None, "model": None}


def configure(transactions=None, node_intel=None, alerts_json=None,
              feedback=None, relay=None, model=None) -> None:
    """`relay` and `model` are the relay matrix and origination model the peer
    profiles read; None means config.yaml's paths."""
    STATE.update({"transactions": transactions, "intel_dir": node_intel,
                  "alerts_json": alerts_json, "feedback": feedback,
                  "relay": relay, "model": model})
    invalidate()


def invalidate() -> None:
    """Forget everything derived from the dataset on disk.

    Called when the artefacts change under us — a red-team injection, or a
    reset — so the next request rebuilds rather than serving the old case.
    """
    for cached in (_transactions, _intel, _features, _profiles, _fingerprints):
        cached.cache_clear()
    BUNDLE.clear()


@lru_cache(maxsize=1)
def _transactions() -> pd.DataFrame:
    return load(STATE["transactions"])


@lru_cache(maxsize=1)
def _intel():
    cfg = config.load()
    return load_intel(None, STATE["intel_dir"] or cfg["ingest"]["input_dir"], cfg)


@lru_cache(maxsize=1)
def _features() -> tuple:
    """The wallet graph and its clustering — needed for entity detail and subgraphs.

    ponytail: rebuilt once per process from the parquet, not incrementally
    maintained. Fine for a demo-sized case; a live deployment would serve this
    from the same store the pipeline writes.
    """
    cfg = config.load()
    graph = build_graph(_transactions(), cfg)
    return graph, FeatureSet.from_graph(graph, cfg)


@lru_cache(maxsize=1)
def _profiles() -> correlation_profile.Sources:
    """The reverse direction's inputs: the hop dataset this API serves, the
    relay matrix, and the origination model's answers over it. Built once."""
    cfg = config.load()
    try:
        df = _transactions()
    except FileNotFoundError:
        df = None
    graph_features = _features()[1] if df is not None else None
    matrix = pd.read_parquet(STATE["relay"]) if STATE["relay"] else None
    model = OriginationModel.load(STATE["model"]) if STATE["model"] else None
    return correlation_profile.build_sources(cfg, df, matrix, model, graph_features, _intel())


@lru_cache(maxsize=1)
def _fingerprints() -> dict[str, dict]:
    """txid -> wallet-construction fingerprint, for every served transaction."""
    return fingerprint.answers_for(iter_transactions(_frame()))


# The full signal bundle: every engine's output for every entity. Built on
# first use (about three seconds on the demo dataset) because only red team
# needs it, and kept mutable so an incremental run can extend it in place.
BUNDLE: dict = {}


def bundle() -> dict:
    if not BUNDLE:
        cfg = config.load()
        df = _frame()
        built = collect_signals(df, cfg, cfg["ingest"]["input_dir"])
        built["df"] = df
        built["intel"] = _intel()
        built["stacker"] = incremental.load_stacker(cfg)
        BUNDLE.update(built)
    return BUNDLE


def commit_bundle(updated: dict) -> None:
    """Take the result of an incremental run as the new truth.

    The updated alerts are written where the console reads them, and the
    cheaper caches are dropped so the queue, the case pages and the graph all
    show the injected pattern without a restart.
    """
    cfg = config.load()
    alerts = updated.get("alerts_frame")
    if alerts is not None:
        parquet = Path(cfg["fusion"]["alerts_parquet"])
        parquet.parent.mkdir(parents=True, exist_ok=True)
        alerts.to_parquet(parquet, index=False)
        payload = {
            "generated_from": "red-team incremental run",
            "alert_threshold": cfg["fusion"]["alert_threshold"],
            "stacker": updated["stacker"].metrics,
            "alerts": json.loads(alerts.to_json(orient="records")),
        }
        Path(cfg["fusion"]["alerts_json"]).write_text(json.dumps(payload, indent=2))
    for cached in (_transactions, _intel, _features, _profiles, _fingerprints):
        cached.cache_clear()
    BUNDLE.clear()
    BUNDLE.update(updated)


def _frame() -> pd.DataFrame:
    try:
        return _transactions()
    except FileNotFoundError:
        raise HTTPException(404, "no processed transactions — run the pipeline first")


def _alerts() -> dict:
    path = Path(STATE["alerts_json"] or config.get("fusion.alerts_json"))
    return json.loads(path.read_text()) if path.exists() else {"alerts": []}


def _alert_rows() -> list[dict]:
    # Leads stay a JSON string on this route (the console's type); each one is
    # given a validity verdict first, so no route serves an origin without one.
    return [{**row, "leads": json.dumps(_leads(row["leads"]))} if row.get("leads") else row
            for row in _alerts().get("alerts", [])]


def _leads(raw) -> list[dict]:
    """A stored alert's leads, each carrying a validity verdict. Leads written
    before the validity layer existed are marked NOT_ASSESSED, never PASS."""
    leads = json.loads(raw) if isinstance(raw, str) else list(raw or [])
    return [lead if "validity" in lead else {**lead, "validity": NOT_ASSESSED.as_dict()}
            for lead in leads]


def _alert_for(entity_id: str) -> dict | None:
    for alert in _alert_rows():
        if entity_id in (alert.get("alert_id"), alert.get("entity_id")):
            return alert
    return None


@app.get("/version")
def version() -> dict:
    """What is actually running here.

    The console fetches this on load and compares `commit` with the one
    compiled into its own bundle; a mismatch means one half is stale and the
    page says so rather than showing yesterday's answers.
    """
    return {
        "commit": _commit(),
        "started_at": STARTED_AT,
        # From the schema, not from app.routes: routes mounted through a
        # router are wrapped and would not be counted. A missing endpoint is
        # exactly the kind of staleness this is here to catch.
        "routes": len(app.openapi().get("paths", {})),
    }


# --- dashboard ------------------------------------------------------------
@app.get("/stats")
def stats() -> dict:
    """Header counts for the dashboard, plus whether origin estimation is degraded."""
    cfg = config.load()
    df = _frame()
    propagation = degraded_mode(df)
    payload = _alerts()
    alerts = payload.get("alerts", [])
    _, features = _features()

    by_pattern: dict[str, int] = {}
    for alert in alerts:
        for pattern in alert.get("pattern_types") or ["(no rule fired)"]:
            by_pattern[pattern] = by_pattern.get(pattern, 0) + 1
    confidences = [float(a["risk_score"]) for a in alerts if a.get("risk_score") is not None]

    return {
        "rows": int(len(df)),
        "transactions": propagation["transactions"],
        "total_entities": len(features.clustering.clusters),
        "total_alerts": len(alerts),
        "alerts_by_pattern_type": dict(sorted(by_pattern.items())),
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "features": {
            "origin_estimation": {
                "status": "degraded" if propagation["degraded"] else "ok",
                "reason": propagation["reason"],
                "multi_hop_transactions": propagation["multi_row_transactions"],
                "mean_observations_per_transaction": propagation["mean_observations"],
                "estimator": cfg["engines"]["propagation"]["estimator"],
            },
        },
        "alerts": len(alerts),          # kept: the dashboard header reads this
        "stacker": payload.get("stacker", {}),
        "intel": _intel().manifest().get("loaded", {}),
    }


def queue_position(entity_id: str) -> dict:
    """This entity's rank in the ranked queue, 1-based, and how long the queue is."""
    rows = sorted(_alert_rows(), key=queue_key)
    for rank, row in enumerate(rows, 1):
        if row.get("entity_id") == entity_id:
            return {"rank": rank, "of": len(rows)}
    return {"rank": None, "of": len(rows)}


def queue_key(row: dict) -> tuple:
    """`fusion.ordering.SORT_KEY` as a tuple a Python sort can use.

    Descending fields are negated rather than the list reversed, because the
    directions differ: risk descends, hops ascend, and the entity id ascends
    last as the deterministic backstop.
    """
    return (
        -float(row.get("risk_score") or 0.0),
        -int(row.get("rule_typologies") or 0),
        int(row["taint_hops"]) if row.get("taint_hops") is not None else NO_TAINT,
        -float(row.get("lead_confidence") or 0.0),
        -int(row.get("tx_count") or 0),
        str(row.get("entity_id") or ""),
    )


@app.get("/alerts")
def alerts(limit: int | None = None, offset: int = Query(0, ge=0),
           min_score: float | None = Query(None, ge=0.0, le=1.0),
           entity_type: str | None = None,
           pattern_type: str | None = None) -> dict:
    """Ranked alerts, highest risk first, paginated and filtered."""
    cfg = config.load()
    size = min(limit or cfg["api"]["page_size"], cfg["api"]["max_page_size"])
    rows = _alert_rows()
    if min_score is not None:
        rows = [r for r in rows if float(r.get("risk_score") or 0.0) >= min_score]
    if entity_type:
        rows = [r for r in rows if r.get("entity_type") == entity_type]
    if pattern_type:
        rows = [r for r in rows if pattern_type in (r.get("pattern_types") or [])]
    # The same order as the queue on screen and the parquet on disk: the
    # composite first, then the published tiebreakers. Filtering a ranked list
    # must not silently re-rank it.
    rows.sort(key=queue_key)
    payload = _alerts()
    return {
        "alert_threshold": payload.get("alert_threshold"),
        "stacker": payload.get("stacker", {}),
        "total": len(rows),
        "limit": size, "offset": offset,
        "filters": {"min_score": min_score, "entity_type": entity_type,
                    "pattern_type": pattern_type},
        "alerts": rows[offset:offset + size],
    }


# --- one entity -----------------------------------------------------------
@app.get("/entities/{entity_id}")
def entity(entity_id: str) -> dict:
    """Everything known about one entity: features, engine scores, explanation."""
    _, features = _features()
    table = features.entities
    row = table[table["cluster_id"] == entity_id]
    if row.empty:
        raise HTTPException(404, f"unknown entity {entity_id}")
    record = json.loads(row.iloc[0].to_json())
    alert = _alert_for(entity_id) or {}
    wallets = sorted(features.clustering.clusters.get(entity_id, {entity_id}))

    return {
        "entity_id": entity_id,
        "entity_type": alert.get("entity_type",
                                 "cluster" if len(wallets) > 1 else "wallet"),
        "wallets": wallets,
        "flag": features.clustering.flags.get(entity_id),
        "features": {k: v for k, v in record.items() if k != "cluster_id"},
        "scores": {
            "risk_score": alert.get("risk_score"),
            "rule_score": alert.get("rule_score"),
            "anomaly_score": alert.get("anomaly_score"),
            "gnn_score": alert.get("gnn_score"),
            "taint_score": alert.get("taint_score"),
            "top_signal": alert.get("top_signal"),
            "contributions": json.loads(alert["contributions"])
            if alert.get("contributions") else {},
        },
        "alerted": bool(alert),
        # Where this case sits in the queue, and the key that put it there —
        # so a case report printed from this record carries the same ordering
        # the analyst saw on screen.
        "queue_position": queue_position(entity_id),
        "sort_key": json.loads(alert["sort_key"]) if alert.get("sort_key") else None,
        "tiebreakers": {k: alert.get(k) for k in TIEBREAKERS} if alert else {},
        "pattern_types": alert.get("pattern_types", []),
        "reason": alert.get("reason"),
        "evidence": alert.get("evidence", []),
        "taint_path": alert.get("taint_path", []),
        "leads": _leads(alert.get("leads")),
        "fingerprints": _entity_fingerprints(set(wallets), features),
        "caveat": ("scores rank leads for a human; an entity without an alert is not "
                   "cleared, only unremarkable"),
    }


def _entity_fingerprints(wallets: set[str], features) -> dict:
    """How the transactions spending this entity's wallets were built, and how
    sure the clustering is that these wallets belong together — a fingerprint
    mismatch lowers that confidence and never made the cluster."""
    answers = _fingerprints()
    spent = sorted({tx.txid for tx in iter_transactions(_frame())
                    if wallets & set(tx.input_addresses)})
    cluster = features.clustering.cluster_of(next(iter(wallets))) if wallets else None
    conflicts = features.clustering.conflicts.get(cluster, [])
    return {**fingerprint.distribution([answers[t] for t in spent if t in answers]),
            "cluster_confidence": features.clustering.confidence.get(cluster, 1.0),
            "conflicts": [{"txid": m["txid"], "heuristic": m["heuristic"],
                           "fingerprints": m["fingerprints"]} for m in conflicts],
            "note": ("fingerprints only lower a merge's confidence when the wallets' "
                     "spending transactions were built differently; they never create one"),
            "console_display": fingerprint.console_display()}


def subgraph(entity_id: str, hops: int) -> dict:
    """Wallets, transactions and IPs within `hops` of this entity's wallets.

    Cytoscape.js elements format. Hops are counted on the wallet/transaction
    graph, so one hop out of a wallet reaches its transactions and two reaches
    the counterparty wallets.
    """
    cfg = config.load()
    graph, features = _features()
    wallets = features.clustering.clusters.get(entity_id)
    if not wallets:
        if entity_id not in graph:
            raise HTTPException(404, f"unknown entity {entity_id}")
        wallets = {entity_id}

    limit = cfg["api"]["max_graph_nodes"]
    seen: dict[str, int] = {w: 0 for w in wallets if w in graph}
    frontier = list(seen)
    for depth in range(1, hops + 1):
        nxt = []
        for node in frontier:
            for neighbour in set(graph.successors(node)) | set(graph.predecessors(node)):
                if neighbour not in seen and len(seen) < limit:
                    seen[neighbour] = depth
                    nxt.append(neighbour)
        frontier = nxt

    def entity_of(addr: str) -> str | None:
        return features.clustering.cluster_of(addr)

    nodes = []
    for node, depth in seen.items():
        data = graph.nodes[node]
        kind = data.get("node_type", WALLET)
        owner = entity_of(node) if kind == WALLET else None
        nodes.append({"data": {
            "id": node, "type": kind, "hop": depth,
            "label": node[:10] + "…" if kind != IP and len(node) > 12 else node,
            "full_id": node,
            "entity_id": owner,
            "is_focus": owner == entity_id or node in wallets,
            **({"asn": data.get("asn"), "country": data.get("geo_country")}
               if kind == IP else {}),
            **({"fee": data.get("fee"), "script_type": data.get("script_type")}
               if kind == TRANSACTION else {}),
        }})

    edges = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        if u in seen and v in seen:
            edges.append({"data": {
                "id": f"{u}|{key}|{v}", "source": u, "target": v,
                "type": data.get("kind"), "label": data.get("kind"),
                "amount": data.get("amount"),
            }})

    return {
        "entity_id": entity_id, "hops": hops,
        "truncated": len(seen) >= limit,
        "counts": {"wallets": sum(1 for n in nodes if n["data"]["type"] == WALLET),
                   "transactions": sum(1 for n in nodes if n["data"]["type"] == TRANSACTION),
                   "ips": sum(1 for n in nodes if n["data"]["type"] == IP)},
        "layout": {"name": "cose"},
        "elements": {"nodes": nodes, "edges": edges},
    }


@app.get("/entities/{entity_id}/graph")
def entity_graph(entity_id: str, hops: int | None = Query(None, ge=1, le=4)) -> dict:
    return subgraph(entity_id, hops or config.get("api.graph_hops"))


@app.get("/entities/{entity_id}/report")
def entity_report(entity_id: str, hops: int | None = Query(None, ge=1, le=4),
                  investigation: str | None = None) -> Response:
    """A one-page PDF case report, ready to attach to a file.

    With `?investigation=<id>` the figure is the analyst's own saved view —
    the same nodes in the same arrangement they were looking at — rather than
    a freshly computed neighbourhood. A case report should show what the
    investigator saw.
    """
    detail = entity(entity_id)
    graph = (_investigation_figure(investigation)
             if investigation else subgraph(entity_id, hops or config.get("api.graph_hops")))
    seal = _evidence_seal()
    pdf = render_pdf(detail, graph, datetime.now(timezone.utc), custody=seal)
    # The report cannot contain its own hash, so it carries the ledger head and
    # the dataset hashes, and the ledger carries the report's hash. Either half
    # identifies the other: a PDF with no matching entry was not produced here.
    entry = custody.record("export.case_report", {
        "files": seal["files"],
        "entity_id": entity_id, "investigation": investigation,
        "risk_score": detail["scores"].get("risk_score"),
        "report_sha256": hashlib.sha256(pdf).hexdigest(),
        "report_bytes": len(pdf), "sealed_at_head": seal["head"]})
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="btc-intel-{entity_id}.pdf"',
        "x-custody-entry": str(entry.get("seq") or ""),
        "x-custody-report-sha256": hashlib.sha256(pdf).hexdigest()})


def _evidence_seal() -> dict:
    """What the data looked like when this export was made."""
    cfg = config.load()
    paths = [Path(cfg["ingest"]["output_path"]), Path(cfg["fusion"]["alerts_json"])]
    return {"head": custody.head(cfg),
            "files": [custody.seal(p) for p in paths if p.exists()]}


def _investigation_figure(investigation_id: str) -> dict:
    """A saved investigation, in the shape the report's figure draws."""
    record = graph_api.load_investigation(investigation_id)
    state = record.get("state", {})
    elements = state.get("elements", [])
    nodes = [e for e in elements if "source" not in e.get("data", {})]
    edges = [e for e in elements if "source" in e.get("data", {})]
    counts = {kind: sum(1 for n in nodes if n["data"].get("type") == kind)
              for kind in ("wallet", "transaction", "ip")}
    return {
        "elements": {"nodes": nodes, "edges": edges},
        "positions": state.get("positions", {}),
        "counts": {"wallets": counts["wallet"], "transactions": counts["transaction"],
                   "ips": counts["ip"]},
        "hops": state.get("hops", 0),
        "truncated": False,
        "source_label": f"saved investigation {record.get('name') or record['id']}",
    }


# --- analyst feedback -----------------------------------------------------
class Feedback(BaseModel):
    status: str      # "confirmed" | "false_positive"


@app.post("/alerts/{alert_id}/feedback")
def alert_feedback(alert_id: str, body: Feedback) -> dict:
    """Record an analyst's verdict, for recalibrating the stacker later.

    Appended, never updated: a changed mind is a second row with a later
    timestamp, because the sequence of verdicts is itself evidence about the
    alert. Nothing here re-fits anything; fusion/stacker.py reads this file when
    it is next trained.
    """
    if body.status not in ("confirmed", "false_positive"):
        raise HTTPException(422, "status must be 'confirmed' or 'false_positive'")
    alert = _alert_for(alert_id)
    if alert is None:
        raise HTTPException(404, f"unknown alert {alert_id}")

    path = Path(STATE["feedback"] or config.get("fusion.feedback_parquet"))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "alert_id": alert_id,
        "entity_id": alert.get("entity_id", alert_id),
        "status": body.status,
        "risk_score": float(alert.get("risk_score") or 0.0),
        "top_signal": alert.get("top_signal"),
        "recorded_at": pd.Timestamp.now(tz="UTC"),
    }])
    # ponytail: read-concat-write. One analyst clicking a button at human speed;
    # if this ever needs concurrency, it wants a real append (or a database).
    if path.exists():
        row = pd.concat([pd.read_parquet(path), row], ignore_index=True)
    row.to_parquet(path, index=False)
    entry = custody.record("verdict", {
        "files": [custody.seal(path)],
        "alert_id": alert_id, "entity_id": alert.get("entity_id", alert_id),
        "status": body.status, "risk_score": float(alert.get("risk_score") or 0.0)})
    return {"alert_id": alert_id, "status": body.status, "recorded": len(row),
            "path": str(path), "custody": {"seq": entry.get("seq")}}


# --- chain of custody -----------------------------------------------------
@app.get("/custody")
def custody_log(limit: int = 200) -> dict:
    """The custody ledger, newest last — what this system did, in order."""
    entries = custody.read()
    return {"total": len(entries), "actor": config.get("custody.actor"),
            "head": custody.head(), "entries": entries[-limit:]}


@app.get("/custody/verify")
def custody_verify(files: bool = True) -> dict:
    """Re-walk the hash chain and re-hash the files it recorded.

    Two separate questions — an edited entry and an edited dataset fail
    differently — so the answer reports them separately. Honest about what it
    cannot prove: see the module docstring in `custody.py`.
    """
    return custody.verify(check_files=files)


# --- propagation ----------------------------------------------------------
@app.get("/transactions/{txid}/propagation")
def propagation(txid: str) -> dict:
    """The observed propagation tree, in Cytoscape elements format.

    The estimated origin is marked `origin`, the runner-ups `runner_up`, and
    every node carries an IP-class badge so the dashboard can show at a glance
    that a candidate is a public relay rather than somebody's home connection.
    """
    df = _frame()
    rows = df[df["txid"] == txid]
    if rows.empty:
        raise HTTPException(404, f"unknown transaction {txid}")

    tree = build_trees(rows)[txid]
    tx = next(iter_transactions(rows), None) if "input_addresses" in rows else None
    cfg = config.load()
    estimate = estimate_origin(tree, _intel(), cfg, tx=tx)
    answer = validity.answer(estimate.ip, estimate.validity, cfg)
    runner_ups = {ip: score for ip, score in estimate.runner_ups[:3]}
    ranked = dict(estimate.ranked)

    nodes = []
    for ip in tree.ips:
        classification = _intel().classify(ip, tree.graph.nodes.get(ip, {}).get("asn"))
        role = ("origin" if ip == estimate.ip
                else "runner_up" if ip in runner_ups else "relay")
        nodes.append({"data": {
            "id": ip, "label": ip, "role": role,
            "ip_class": classification.ip_class,
            "badge": CLASS_BADGE.get(classification.ip_class, classification.ip_class),
            "asn": classification.asn,
            "score": round(float(ranked.get(ip, 0.0)), 6),
            "first_seen": tree.first_seen.get(ip),
            "evidence": classification.evidence,
        }})
    edges = [{"data": {"id": f"{u}->{v}", "source": u, "target": v,
                       "timestamp": data["timestamp"]}}
             for u, v, data in tree.graph.edges(data=True)]

    return {
        "txid": txid,
        "estimated_origin": estimate.ip,
        "ip_class": estimate.ip_class,
        "confidence": estimate.confidence,
        "attribution_confidence": estimate.attribution_confidence,
        "estimator": estimate.estimator,
        "degraded": estimate.degraded,
        "low_confidence_origin": estimate.low_confidence,
        "anonymized_entry_point": estimate.anonymized_entry_point,
        "probability": estimate.confidence,
        "calibration_basis": CALIBRATION_BASIS,
        # PASS, or the reason the origin is withheld and the evidence for it.
        # An ABSTAIN-tier verdict also sets low_confidence_origin.
        "validity": estimate.validity.as_dict(),
        # What the answer may claim: an IP attribution, an onion identity (no
        # IP), or a CoinJoin's broadcasting peer (no input ownership). None
        # when the verdict withholds it.
        "answer": answer,
        "n_observations": estimate.n_observations,
        "runner_ups": [{"ip": ip, "score": round(float(s), 6)} for ip, s in runner_ups.items()],
        "caveat": ("estimated origin is a probabilistic lead, not an attribution — "
                   "the true origin is often absent from the observed hops"
                   + ("; this candidate is an anonymized entry point (Tor exit or "
                      "hosting), which is where the broadcast entered the network, "
                      "not who sent it" if estimate.anonymized_entry_point else "")),
        "layout": {"name": "dagre", "roots": [estimate.ip] if estimate.ip else []},
        "elements": {"nodes": nodes, "edges": edges},
    }


@app.get("/transactions/{txid}/fingerprint")
def transaction_fingerprint(txid: str) -> dict:
    """How this transaction was likely built: a ranked set of construction
    patterns, or unknown. Not which software built it, and never a party."""
    answer = _fingerprints().get(txid)
    if answer is None:
        raise HTTPException(404, f"transaction {txid} has no structure in the served dataset")
    return {"txid": txid, **answer, "console_display": fingerprint.console_display()}


@app.get("/transactions/{txid}/origination")
def origination(txid: str) -> dict:
    """The origination model's answer for this txid, once per capture it was
    seen in. The same frame a peer profile's `originated` reads, so every
    transaction a profile claims shows the same peer here."""
    answers = correlation_profile.transaction_origination(txid, _profiles())
    return {"txid": txid, "captures": answers,
            "note": None if answers else "no capture in the relay matrix contains this txid"}


# --- the reverse direction: peer -> profile -------------------------------
def _lookup_record(kind: str, subject: str, profile: dict | None) -> dict:
    """Every profile lookup goes in the custody ledger, found or not: which
    peers an investigation asked about is itself part of the record."""
    return custody.record(f"lookup.{kind}_profile", {
        "subject": subject, "found": profile is not None,
        "simulated_only": (profile or {}).get("header", {}).get("simulated_only"),
        "files": _evidence_seal()["files"]})


@app.get("/peers/{peer}/profile")
def peer_profile(peer: str) -> dict:
    """What one peer — an IP or an onion identity — did on the network."""
    peer = peer.strip()
    profile = correlation_profile.peer_profile(peer, _profiles())
    entry = _lookup_record("peer", peer, profile)
    if profile is None:
        raise HTTPException(404, f"peer {peer} is not in any capture or the relay-hop dataset")
    return {**profile, "custody": {"seq": entry.get("seq")}}


# --- actors ---------------------------------------------------------------
def _actors() -> dict:
    """fusion/actors.py's output, written beside the alerts it was built with."""
    from fusion.pipeline import actors_path_for
    cfg = config.load()
    path = actors_path_for(Path(STATE["alerts_json"] or cfg["fusion"]["alerts_json"]), cfg)
    return json.loads(path.read_text()) if path.exists() else {"actors": []}


def _actor(actor_id: str) -> dict | None:
    return next((a for a in _actors()["actors"] if a["actor_id"] == actor_id), None)


def _drill_down(actor: dict) -> dict:
    """Where each part of an actor is evidenced, in the views that already exist."""
    return {"entities": {m: f"/entities/{m}" for m in actor["members"]},
            "peers": {p: f"/peers/{p}/profile" for p in actor["peers"]},
            "transactions": sorted({e["txid"] for lk in actor["links"]
                                    for e in lk["evidence"]})}


@app.get("/actors")
def actors_queue(limit: int | None = None, offset: int = Query(0, ge=0),
                 alerted_only: bool = True) -> dict:
    """The actor queue: clusters joined to peer identities (docs/ACTORS.md),
    highest risk first. `alerted_only=false` lists every actor."""
    cfg = config.load()
    size = min(limit or cfg["api"]["page_size"], cfg["api"]["max_page_size"])
    payload = _actors()
    rows = [a for a in payload["actors"] if a["alerted"] or not alerted_only]
    return {"alert_threshold": payload.get("alert_threshold"),
            "statement": payload.get("statement"), "total": len(rows),
            "limit": size, "offset": offset, "actors": rows[offset:offset + size]}


@app.get("/actors/{actor_id}")
def actor_detail(actor_id: str) -> dict:
    """One actor with its members, linked peers (basis, confidence, validity
    tiers) and drill-down to the per-entity, per-peer and per-transaction views.
    Every view is recorded in the custody ledger, found or not."""
    actor = _actor(actor_id)
    entry = custody.record("actor.view", {"subject": actor_id, "found": actor is not None,
                                          "files": _evidence_seal()["files"]})
    if actor is None:
        raise HTTPException(404, f"unknown actor {actor_id}")
    return {**actor, "drill_down": _drill_down(actor), "custody": {"seq": entry.get("seq")}}


@app.post("/actors/{actor_id}/verdict")
def actor_verdict(actor_id: str, body: Feedback) -> dict:
    """An analyst's verdict on an actor, appended beside the entity feedback
    (never into it: the stacker trains on entity verdicts) and recorded in the
    custody ledger."""
    if body.status not in ("confirmed", "false_positive"):
        raise HTTPException(422, "status must be 'confirmed' or 'false_positive'")
    actor = _actor(actor_id)
    if actor is None:
        raise HTTPException(404, f"unknown actor {actor_id}")
    entity_feedback = Path(STATE["feedback"] or config.get("fusion.feedback_parquet"))
    path = entity_feedback.with_name("actor_feedback.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"actor_id": actor_id, "members": json.dumps(actor["members"]),
                         "peers": json.dumps(actor["peers"]), "status": body.status,
                         "risk_score": float(actor["risk_score"]),
                         "recorded_at": pd.Timestamp.now(tz="UTC")}])
    if path.exists():
        row = pd.concat([pd.read_parquet(path), row], ignore_index=True)
    row.to_parquet(path, index=False)
    entry = custody.record("actor.verdict", {
        "files": [custody.seal(path)], "actor_id": actor_id, "status": body.status,
        "members": actor["members"], "peers": actor["peers"],
        "risk_score": float(actor["risk_score"])})
    return {"actor_id": actor_id, "status": body.status, "recorded": len(row),
            "custody": {"seq": entry.get("seq")}}


@app.get("/asns/{asn}/profile")
def asn_profile(asn: str) -> dict:
    """Every peer seen in an ASN, aggregated, with the per-peer breakdown."""
    digits = asn.upper().removeprefix("AS")
    if not digits.isdigit():
        raise HTTPException(422, "an ASN is a number, optionally prefixed AS")
    profile = correlation_profile.asn_profile(int(digits), _profiles())
    entry = _lookup_record("asn", f"AS{int(digits)}", profile)
    if profile is None:
        raise HTTPException(404, f"no peer seen in AS{int(digits)}")
    return {**profile, "custody": {"seq": entry.get("seq")}}


# The investigation graph endpoints, sharing this module's cached graph and
# alert list rather than rebuilding either.
graph_api.register(app, _features, _alerts)

# Red team shares the same cached bundle, and hands back an updated one so the
# rest of the console sees an injected pattern immediately.
redteam_api.register(app, bundle, commit_bundle, invalidate)

# Live monitoring folds arriving files into that same bundle, so it is handed
# the same accessors — and takes the same lock (fusion.incremental.LOCK).
monitor_api.register(app, bundle, commit_bundle)


# --- the console itself ----------------------------------------------------
# One process serves the API and the built front end, so an offline demo needs
# no second web server and no npm on the target machine.
#
# The console lives under /app/ rather than at the root because three of its
# routes — /alerts, /entities/{id}, /custody — are also API paths. Sharing the
# root would mean a hard refresh on the alert queue returned JSON to a browser
# that asked for a page. The prefix removes that whole class of collision;
# vite's `base` and the router's `basename` are the same string.
class SinglePageApp(StaticFiles):
    """Static files, with a client-side router behind them.

    The browser asks for /app/entities/bc1q…; there is no such file, because
    that path only means something once React is running. Anything that is not
    a file is answered with index.html, and the app takes it from there. A
    missing *asset* still 404s — that is a broken build, and hiding it behind
    the index page would turn it into a blank screen with no error.
    """

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as missing:
            # Starlette raises rather than returning, so the fallback has to be
            # an except clause. A path with a file extension is an asset that
            # should be there: let that 404 stand, because answering a missing
            # bundle with index.html turns a broken build into a blank page.
            if missing.status_code != 404 or Path(path).suffix:
                raise
            return await super().get_response("index.html", scope)


DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if (DIST / "index.html").exists():
    app.mount("/app", SinglePageApp(directory=DIST, html=True), name="console")

    @app.get("/", include_in_schema=False)
    def console() -> RedirectResponse:
        """The address people are given. Everything else is under /app/."""
        return RedirectResponse("/app/")
else:                                        # a dev checkout that has not built
    @app.get("/", include_in_schema=False)
    def console_missing() -> dict:
        return {"console": "not built",
                "detail": f"no {DIST}/index.html — run `npm --prefix web run build`, "
                          "or use the dev server on :5173 while developing",
                "api": "/docs"}
