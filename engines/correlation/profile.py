"""The correlation engine, run backwards: peer -> behavioural profile.

Forward, this package answers "which peer broadcast for this cluster?" from
transaction origins. Reverse, it answers "what did this peer do on the
network?" from the same evidence, for a peer named by IP or onion identity, or
for every peer of an ASN at once:

  originated  transactions the origination model (origination/) names this peer
              as origin of, each with its calibrated probability and validity
              tier. QUALIFIED and ANNOTATE answers stay qualified and annotated;
              an ABSTAIN is listed apart and never counted as originated.
  relayed     transactions it announced without being named origin: a count and
              a sample, not a dump.
  timing      announce rate, active hours (UTC), inter-announcement statistics —
              or a sentence saying there are too few observations for them.
  clients     user agent and service-flag history, first and last seen.
  clusters    wallet clusters (graph/) linked through origination, through a
              correlation lead, or both; each with its confidence and its
              evidence chain back to the rows it was built from.
  network     ASN and country from ingest/ enrichment. Never on an onion
              identity, which has no IP to enrich (P6.1's TOR_ONION rule).
  vantage     every capture and observer the peer was seen from.

WORDING IS PART OF THE CONTRACT. A profile describes a network peer's observed
behaviour. It says "peer X" and "cluster C" and never who operates either: an
IP is a network location that many people may share, and a cluster is a set of
addresses that co-spend. tests/test_profile.py asserts this on every profile.

Two data sources feed a profile, and each says where it came from:
  * the relay matrix (`features.relay`, one row per candidate peer per txid per
    capture), scored by the origination model — per-capture and single-vantage;
  * the ingested relay-hop dataset (`ingest.output_path`), which the forward
    engine already correlates — its origins are `engines.propagation`'s.
A profile built only from simulated or fixture data says so in its header.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pandas as pd

import config
from analysis import validity
from analysis.validity import COINJOIN, Verdict
from eval.origin import abstention_reasons
from features import fingerprint
from graph.builder import Tx, iter_transactions
from graph.clustering import is_coinjoin

from .scorer import (
    Observation,
    collect_observations,
    raw_confidence,
    score_observations,
)

HOP_SOURCE = "relay-hop dataset"
MATRIX_SOURCE = "relay matrix"
CAVEAT = ("A profile of network behaviour. It describes what peer {peer} announced, "
          "as seen from the observers listed, and says nothing about who operates it: "
          "an address can be shared, reassigned or a relay for others, and every "
          "origin in it is a calibrated estimate, not an attribution.")


# --- provenance ------------------------------------------------------------
def capture_provenance(capture_source: str | None) -> str:
    """fixture | simulated | capture. The reader stamps every row with where it
    came from; only the two synthetic prefixes are recognised as synthetic, and
    anything else is a capture file in a real format."""
    source = str(capture_source or "")
    if source.startswith("synthetic-fixture:"):
        return "fixture"
    if source.startswith("sim:"):
        return "simulated"
    return "capture"


def is_onion(peer: str) -> bool:
    return str(peer).endswith(".onion")


def _iso(ts) -> str | None:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.tz_convert("UTC").isoformat() if ts.tzinfo else ts.isoformat()
    return datetime.fromtimestamp(float(ts), UTC).isoformat()


def _epoch(ts) -> float:
    return pd.Timestamp(ts).timestamp() if not isinstance(ts, (int, float)) else float(ts)


# --- the forward answer, shared --------------------------------------------
def _shapes(matrix: pd.DataFrame, txs: dict[str, Tx]) -> pd.DataFrame | None:
    """Transaction structure for the matrix's txids that the chain data holds,
    so the COINJOIN detector sees it — exactly as the TXID page does."""
    rows = []
    for capture_id, txid in matrix[["capture_id", "txid"]].drop_duplicates().itertuples(
            index=False):
        tx = txs.get(txid)
        if tx is not None:
            rows.append({"capture_id": capture_id, "txid": txid,
                         "in_addrs": [a for a, _ in tx.inputs],
                         "in_vals": [v for _, v in tx.inputs],
                         "out_addrs": [a for a, _ in tx.outputs],
                         "out_vals": [v for _, v in tx.outputs]})
    return pd.DataFrame(rows) if rows else None


def _verdict(row) -> Verdict:
    return Verdict(row.validity, None if pd.isna(row.validity_confidence)
                   else float(row.validity_confidence),
                   tuple(row.validity_evidence), tuple(row.validity_reasons))


def origination_answers(model, matrix: pd.DataFrame, cfg: dict,
                        txs: dict[str, Tx] | None = None,
                        shapes: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (capture_id, txid): the peer the origination model names,
    its calibrated probability, the validity verdict, and whether the answer is
    given — by `eval.origin.flagged_at`'s rule at the model's own cutoff.

    Both directions read this frame: the TXID page's per-capture origination
    and a peer profile's `originated`. That is what makes the round trip hold.
    """
    shapes = shapes if shapes is not None else _shapes(matrix, txs or {})
    decided = model.decide(matrix, None, cfg, shapes)
    if decided.empty:
        return decided
    decided["abstention_reason"] = abstention_reasons(decided, cfg, model.cutoff)
    decided["answered"] = decided["abstention_reason"].isna()
    decided["answer"] = [validity.answer(row.estimated_origin_ip if row.answered else None,
                                         _verdict(row), cfg)
                         for row in decided.itertuples()]
    first = matrix.sort_values("announce_ts").groupby(["capture_id", "txid"]).first()
    key = pd.MultiIndex.from_frame(decided[["capture_id", "txid"]])
    decided["observer"] = first["observer_ip"].reindex(key).to_numpy()
    decided["capture_source"] = first["capture_source"].reindex(key).to_numpy()
    return decided


# --- sources ---------------------------------------------------------------
@dataclass
class Sources:
    """Everything a profile reads, built once and then queried per peer."""

    matrix: pd.DataFrame | None = None           # features.relay rows
    answers: pd.DataFrame | None = None          # origination_answers()
    transactions: pd.DataFrame | None = None     # ingest output: relay-hop rows
    features: object | None = None               # engines.rules FeatureSet
    origins: pd.DataFrame | None = None          # engines.propagation, per hop txid
    observations: list[Observation] = field(default_factory=list)
    leads: pd.DataFrame | None = None            # score_observations(observations)
    txs: dict[str, Tx] = field(default_factory=dict)
    intel: object | None = None                  # ingest.ip_intel, for ip_class
    fingerprints: dict = field(default_factory=dict)  # txid -> features.fingerprint answer
    hop_provenance: str = "simulated"
    model_note: str | None = None                # why there are no answers, if none
    _index: dict = field(default_factory=dict, repr=False)

    def rows(self, name: str, column: str, peer: str) -> pd.DataFrame | None:
        """`getattr(self, name)`'s rows whose `column` is `peer`, or None.
        Grouped once per frame, so profiling every peer stays linear."""
        frame = getattr(self, name)
        if frame is None or not len(frame):
            return None
        if name not in self._index:
            self._index[name] = frame.groupby(frame[column].astype(str), sort=False).indices
        positions = self._index[name].get(peer)
        return None if positions is None else frame.iloc[positions]


def build_sources(cfg: dict | None = None, transactions: pd.DataFrame | None = None,
                  matrix: pd.DataFrame | None = None, model=None, features=None,
                  intel=None) -> Sources:
    """Anything not handed in is read from where config.yaml says it lives."""
    from engines.propagation.estimators import estimate_all
    from engines.rules.detectors import FeatureSet
    from graph.builder import build_graph, load
    from ingest.ip_intel import load_intel
    from origination.model import OriginationModel

    cfg = cfg or config.load()
    src = Sources(hop_provenance=cfg["ingest"].get("provenance", "simulated"))
    if transactions is None and Path(cfg["ingest"]["output_path"]).exists():
        transactions = load(None, cfg)
    if transactions is not None and len(transactions):
        intel = intel if intel is not None else load_intel(cfg=cfg)
        features = features or FeatureSet.from_graph(build_graph(transactions, cfg), cfg)
        src.transactions, src.features, src.intel = transactions, features, intel
        src.txs = {tx.txid: tx for tx in iter_transactions(transactions)}
        src.origins, _ = estimate_all(transactions, intel, cfg)
        src.observations = collect_observations(transactions, features, cfg, src.origins,
                                                intel)
        src.leads = score_observations(src.observations, cfg)
        src.fingerprints = fingerprint.answers_for(src.txs.values(), cfg)

    relay_path = Path(cfg["features"]["relay_path"])
    if matrix is None and relay_path.exists():
        matrix = pd.read_parquet(relay_path)
    if matrix is not None and len(matrix):
        src.matrix = matrix
        model_path = Path(cfg["origination"]["model_path"])
        if model is None and model_path.exists():
            model = OriginationModel.load(model_path)
        if model is None:
            src.model_note = (f"no origination model at {model_path}; run "
                              "`python -m origination.pipeline evaluate`")
        else:
            src.answers = origination_answers(model, matrix, cfg, src.txs)
    return src


# --- the TXID side ---------------------------------------------------------
def transaction_origination(txid: str, src: Sources) -> list[dict]:
    """The origination model's answer for one txid, per capture it appears in."""
    if src.answers is None or src.answers.empty:
        return []
    rows = src.answers[src.answers["txid"] == txid]
    return [{
        "capture_id": r.capture_id, "observer": r.observer,
        "capture_source": r.capture_source,
        "provenance": capture_provenance(r.capture_source),
        "named_peer": r.estimated_origin_ip, "probability": float(r.confidence),
        "calibration_basis": r.calibration_basis,
        "validity": _verdict(r).as_dict(), "answered": bool(r.answered),
        "abstention_reason": r.abstention_reason, "answer": r.answer,
        "n_candidates": int(r.n_candidates),
    } for r in rows.itertuples()]


# --- one peer ----------------------------------------------------------------
def _claim(r, peer: str) -> dict:
    """One originated transaction, in the shape its answer allows."""
    verdict = _verdict(r)
    if r.answer["kind"] == "broadcasting_peer":
        says = (f"peer {peer} is the broadcasting peer of CoinJoin {r.txid}; the "
                "inputs' ownership is not attributable")
    elif r.answer["kind"] == "onion_identity":
        says = f"onion identity {peer} named as origin of {r.txid}; not for IP-level follow-up"
    else:
        says = f"peer {peer} named as estimated origin of {r.txid}"
    if verdict.tier == validity.ANNOTATE:
        says += f", flagged {', '.join(verdict.reasons)}"
    return {"txid": r.txid, "capture_id": r.capture_id, "observer": r.observer,
            "probability": round(float(r.confidence), 4),
            "calibration_basis": r.calibration_basis, "tier": verdict.tier,
            "validity": verdict.as_dict(), "answer": r.answer, "statement": says}


def _vantage(peer: str, mine: pd.DataFrame | None, hops: pd.DataFrame | None) -> list[dict]:
    out = []
    if mine is not None:
        for (capture_id, source), g in mine.groupby(["capture_id", "capture_source"]):
            out.append({"source": MATRIX_SOURCE, "capture_id": capture_id,
                        "capture_source": source, "provenance": capture_provenance(source),
                        "observer": sorted(set(g["observer_ip"].astype(str))),
                        "direction": sorted(set(g["direction"].astype(str))),
                        "vantage": "single observer: the node that wrote the capture",
                        "announcements": len(g),
                        "first_seen": _iso(g["announce_ts"].min()),
                        "last_seen": _iso(g["announce_ts"].max())})
    if hops is not None and len(hops):
        out.append({"source": HOP_SOURCE, "capture_id": None, "capture_source": None,
                    "provenance": None,           # filled from Sources by the caller
                    "observer": sorted(set(hops["dst_ip"].astype(str))),
                    "direction": ["sent"],
                    "vantage": ("multi-point relay log: each row is one send from peer "
                                f"{peer}, and the observers are the receiving nodes"),
                    "announcements": int(hops["txid"].nunique()),
                    "first_seen": _iso(hops["timestamp"].min()),
                    "last_seen": _iso(hops["timestamp"].max())})
    return out


def timing_signature(peer: str, events: list[tuple[str, float]], cfg: dict) -> dict:
    """`events` are (vantage, epoch seconds), one per announced transaction.
    Intervals are taken within a vantage only: two observers' clocks and views
    are not one timeline."""
    need = cfg["engines"]["correlation"]["profile"]["min_timing_observations"]
    n = len(events)
    if n < need:
        return {"sufficient": False, "announcements": n, "threshold": need,
                "statement": (f"peer {peer} was seen announcing {n} transaction"
                              f"{'' if n == 1 else 's'}; a timing signature needs at least "
                              f"{need} (engines.correlation.profile.min_timing_observations), "
                              "so none is given")}
    by_vantage: dict[str, list[float]] = defaultdict(list)
    for vantage, ts in events:
        by_vantage[vantage].append(ts)
    gaps, minutes = [], 0.0
    for times in by_vantage.values():
        times.sort()
        gaps += [b - a for a, b in pairwise(times)]
        minutes += (times[-1] - times[0]) / 60.0
    hours = Counter(datetime.fromtimestamp(ts, UTC).hour for _, ts in events)
    return {
        "sufficient": True, "announcements": n, "threshold": need,
        "announce_rate_per_min": round(n / minutes, 4) if minutes > 0 else None,
        "active_hours_utc": [hours.get(h, 0) for h in range(24)],
        "inter_announcement_s": {
            "mean": round(statistics.fmean(gaps), 3) if gaps else None,
            "median": round(statistics.median(gaps), 3) if gaps else None,
            "stdev": round(statistics.stdev(gaps), 3) if len(gaps) > 1 else None,
            "min": round(min(gaps), 3) if gaps else None,
            "max": round(max(gaps), 3) if gaps else None},
        "vantages": len(by_vantage),
        "statement": (f"over {n} announcements from {len(by_vantage)} vantage"
                      f"{'' if len(by_vantage) == 1 else 's'}; intervals are measured "
                      "within one vantage, never across two"),
    }


def _history(mine: pd.DataFrame | None, column: str) -> list[dict]:
    if mine is None or column not in mine:
        return []
    known = mine[mine[column].notna()]
    out = []
    for value, g in known.groupby(column):
        out.append({"value": int(value) if column == "services" else str(value),
                    **({"hex": hex(int(value))} if column == "services" else {}),
                    "first_seen": _iso(g["announce_ts"].min()),
                    "last_seen": _iso(g["announce_ts"].max()),
                    "captures": sorted(set(g["capture_id"]))})
    return sorted(out, key=lambda h: h["first_seen"] or "")


def _raw_matrix_row(row) -> dict:
    return {"source": MATRIX_SOURCE, "capture_id": row.capture_id, "txid": row.txid,
            "peer": row.peer_ip, "announce_ts": _iso(row.announce_ts),
            "capture_source": row.capture_source}


def _raw_hop_row(index, row) -> dict:
    return {"source": HOP_SOURCE, "row": int(index), "txid": row.txid,
            "timestamp": _iso(row.timestamp), "src": str(row.src_ip), "dst": str(row.dst_ip)}


def _linked_clusters(peer: str, claims: list, src: Sources, cfg: dict,
                     mine: pd.DataFrame | None) -> tuple[list[dict], list[dict]]:
    """Clusters peer X is linked to, by basis, with the rows behind each link."""
    limit = cfg["engines"]["correlation"]["profile"]["evidence_rows"]
    links: dict[str, dict] = {}
    excluded = []

    def link(entity):
        return links.setdefault(entity, {"origination": [], "lead": None, "evidence": []})

    # Origination: a claimed origin of a transaction links the peer to the
    # clusters its inputs belong to — never through a CoinJoin, whose inputs
    # have many owners, whether the verdict or the structure says so.
    for claim in claims:
        tx = src.txs.get(claim["txid"])
        if claim["answer"]["kind"] == "broadcasting_peer" or COINJOIN in claim["validity"][
                "reasons"] or (tx is not None and is_coinjoin(tx, cfg)):
            excluded.append({"txid": claim["txid"], "reason": COINJOIN,
                             "statement": ("CoinJoin: peer is the broadcasting peer only; "
                                           "input ownership is not attributable, so no "
                                           "cluster link is drawn")})
            continue
        if tx is None or src.features is None:
            excluded.append({"txid": claim["txid"], "reason": "NO_STRUCTURE",
                             "statement": ("the transaction's inputs are not in the chain "
                                           "data, so no cluster can be linked")})
            continue
        rows = [] if mine is None else [
            _raw_matrix_row(r) for r in mine[(mine["txid"] == claim["txid"]) & (
                mine["capture_id"] == claim["capture_id"])].itertuples()]
        by_entity = defaultdict(list)
        for addr, _ in tx.inputs:
            by_entity[src.features.entity_of(addr)].append(addr)
        for entity, addrs in by_entity.items():
            entry = link(entity)
            entry["origination"].append(claim["probability"])
            entry["evidence"].append({"basis": "origination", "txid": claim["txid"],
                                      "probability": claim["probability"],
                                      "tier": claim["tier"],
                                      "reasons": list(claim["validity"]["reasons"]),
                                      "rows": rows,
                                      "inputs": sorted(set(addrs))})

    # Correlation leads: the forward engine's (cluster, IP) associations.
    if src.leads is not None and len(src.leads):
        hops = src.transactions
        for lead in src.leads[src.leads["ip"] == peer].itertuples():
            entry = link(lead.entity_id)
            entry["lead"] = {"final_score": round(float(lead.final_score), 4),
                             "reason": lead.reason, "validity": lead.validity,
                             "validity_evidence": lead.validity_evidence}
            for obs in src.observations:
                if obs.ip != peer or obs.entity_id != lead.entity_id:
                    continue
                sent = hops[(hops["txid"] == obs.txid) & (hops["src_ip"] == peer)]
                tx = src.txs.get(obs.txid)
                entry["evidence"].append({
                    "basis": "correlation lead", "txid": obs.txid,
                    "probability": round(obs.origin_confidence, 4),
                    "rows": [_raw_hop_row(i, r) for i, r in zip(sent.index,
                                                                 sent.itertuples())],
                    "inputs": sorted({a for a, _ in (tx.inputs if tx else [])
                                      if src.features.entity_of(a) == lead.entity_id})})

    out = []
    for entity, entry in links.items():
        by_basis = {}
        if entry["origination"]:
            by_basis["origination"] = {
                "confidence": round(raw_confidence(sum(entry["origination"]), cfg), 4),
                "transactions": len(entry["origination"]),
                "rule": "1 - exp(-sum of calibrated probabilities / saturation_k)"}
        if entry["lead"]:
            by_basis["correlation lead"] = {"confidence": entry["lead"]["final_score"],
                                            **entry["lead"]}
        basis = "both" if len(by_basis) == 2 else next(iter(by_basis))
        evidence = sorted(entry["evidence"], key=lambda e: (-e["probability"], e["txid"]))
        out.append({
            "cluster": f"cluster {entity}", "cluster_id": entity, "basis": basis,
            "confidence": max(b["confidence"] for b in by_basis.values()),
            "confidence_rule": ("the stronger basis; origination and a correlation lead "
                                "read overlapping evidence, so they are not combined"
                                if basis == "both" else f"the {basis} confidence"),
            "by_basis": by_basis,
            "statement": (f"peer {peer} is linked to cluster {entity} by {basis}: it is "
                          "estimated to have broadcast transactions spending that "
                          "cluster's inputs — an association to investigate"),
            "evidence_chain": "raw rows -> transaction -> input addresses -> cluster",
            "evidence": evidence[:limit], "evidence_total": len(evidence)})
    out.sort(key=lambda c: (-c["confidence"], c["cluster_id"]))
    return out, excluded


def peer_profile(peer: str, src: Sources, cfg: dict | None = None) -> dict | None:
    """Everything the evidence says about one peer, or None if it is in no source."""
    cfg = cfg or config.load()
    p = cfg["engines"]["correlation"]["profile"]
    onion = is_onion(peer)
    mine = src.rows("matrix", "peer_ip", peer)
    hops = None if onion else src.rows("transactions", "src_ip", peer)
    named = src.rows("answers", "estimated_origin_ip", peer)
    if mine is None and hops is None and named is None:
        return None

    # Originated, and the answers withheld — per origination/.
    claims, withheld = [], []
    if named is not None:
        for r in named.sort_values(["capture_id", "txid"]).itertuples():
            if r.answered:
                claims.append(_claim(r, peer))
            else:
                withheld.append({"txid": r.txid, "capture_id": r.capture_id,
                                 "tier": _verdict(r).tier, "reason": r.abstention_reason,
                                 "probability": round(float(r.confidence), 4),
                                 "statement": (f"peer {peer} ranked first for {r.txid}, but "
                                               f"the answer is withheld: "
                                               f"{r.abstention_reason}")})
    tiers = Counter(c["tier"] for c in claims)

    # Relayed: announced, not named origin.
    relayed, named_keys = [], {(c["capture_id"], c["txid"]) for c in claims + withheld}
    if mine is not None:
        for r in mine.sort_values(["capture_id", "announce_ts"]).itertuples():
            if (r.capture_id, r.txid) not in named_keys:
                relayed.append({"source": MATRIX_SOURCE, "txid": r.txid,
                                "capture_id": r.capture_id, "rank": int(r.announce_rank),
                                "candidates": int(r.candidate_count)})
    propagation_named, propagation_txids = 0, []
    if hops is not None and src.origins is not None:
        origin_of = dict(zip(src.origins["txid"], src.origins["estimated_origin_ip"]))
        for txid in hops.sort_values("timestamp")["txid"].drop_duplicates():
            if origin_of.get(txid) != peer:
                relayed.append({"source": HOP_SOURCE, "txid": txid, "capture_id": None,
                                "rank": None, "candidates": None})
            else:
                propagation_named += 1
                propagation_txids.append(txid)

    events = [] if mine is None else [(f"capture:{r.capture_id}", float(r.announce_ts))
                                      for r in mine.itertuples()]
    if hops is not None:
        first = hops.sort_values("timestamp").drop_duplicates("txid")
        events += [(HOP_SOURCE, _epoch(ts)) for ts in first["timestamp"]]

    vantage = _vantage(peer, mine, hops)
    for v in vantage:
        if v["source"] == HOP_SOURCE:
            v["provenance"] = src.hop_provenance
    provenances = sorted({v["provenance"] for v in vantage})
    simulated_only = bool(provenances) and all(v in ("simulated", "fixture")
                                               for v in provenances)

    clusters, excluded = _linked_clusters(peer, claims, src, cfg, mine)
    agents = _history(mine, "user_agent")
    services = _history(mine, "services")

    profile = {
        "subject": f"{'onion identity' if onion else 'peer'} {peer}",
        "peer": peer, "kind": "onion_identity" if onion else "ip",
        "header": {
            "simulated_only": simulated_only, "provenance": provenances,
            "statement": (f"Built only from {' and '.join(provenances)} data: it describes "
                          "the simulator's or a test fixture's peers, not the Bitcoin network"
                          if simulated_only else
                          "Includes data from captures in a real format; each vantage "
                          "below states its own provenance"),
        },
        "originated": {
            "claimed": len(claims),
            "by_tier": {t: tiers.get(t, 0) for t in (validity.PASS, validity.QUALIFIED,
                                                     validity.ANNOTATE)},
            "claims": claims,
            "basis": ("origination model (origination/), per capture; "
                      + (src.model_note or "an answer is claimed only when "
                         "eval.origin.flagged_at does not withhold it")),
        },
        # Hop-dataset transactions whose origin engines.propagation estimates as
        # this peer. Not origination/ claims: they reach a profile only through
        # the correlation leads under linked_clusters, and each one's own TXID
        # page shows the estimate with its verdict.
        "propagation_origin": {
            "count": propagation_named,
            "statement": (f"{propagation_named} relay-hop transaction"
                          f"{'' if propagation_named == 1 else 's'} for which "
                          f"engines.propagation estimates peer {peer} as origin; they "
                          "feed the correlation leads and are not origination claims")},
        "withheld": {"count": len(withheld),
                     "by_reason": dict(Counter(w["reason"] for w in withheld)),
                     "items": withheld},
        "relayed": {"count": len(relayed),
                    "by_source": dict(Counter(r["source"] for r in relayed)),
                    "sample": relayed[:p["relayed_sample"]]},
        "timing": timing_signature(peer, events, cfg),
        "clients": {
            "user_agents": agents, "services": services,
            "statement": None if agents or services else (
                "no version handshake for this peer in any capture: the relay-hop "
                "dataset carries none, and a debug.log records no service bits"),
        },
        "linked_clusters": clusters,
        "excluded_links": excluded,
        "fingerprints": _fingerprints(claims, propagation_txids, src),
        "vantage": vantage,
        "caveat": CAVEAT.format(peer=peer),
    }
    if not onion:
        profile["network"] = _network(peer, mine, hops, src.intel)
    return profile


def _fingerprints(claims: list[dict], propagation_txids: list[str], src: Sources) -> dict:
    """How the transactions this peer originated were built: construction
    patterns, never which software or who. Only transactions whose structure is
    in the chain data can be fingerprinted; the rest are counted, not guessed."""
    def split(txids):
        known = [src.fingerprints[t] for t in txids if t in src.fingerprints]
        return {**fingerprint.distribution(known), "without_structure": len(txids) - len(known)}
    return {"originated": split(list(dict.fromkeys(c["txid"] for c in claims))),
            "propagation_origin": split(propagation_txids),
            "statement": ("construction fingerprints of the transactions this peer is "
                          "named origin of (features/fingerprint.py): a construction "
                          "pattern per transaction, or unknown. " + fingerprint.STATEMENT),
            "console_display": fingerprint.console_display()}


def _network(peer: str, mine: pd.DataFrame | None, hops: pd.DataFrame | None,
             intel=None) -> dict:
    """ASN and country from ingest/ enrichment: `ingest.geoip` for the matrix,
    the dataset's own fields (or GeoIP) for hop rows. First non-null wins."""
    def first(frame, column):
        if frame is None or column not in frame:
            return None
        values = frame[column].dropna()
        return values.iloc[0] if len(values) else None

    asn = first(mine, "asn") if first(mine, "asn") is not None else first(hops, "asn")
    asn = int(asn) if asn is not None else None
    ip_class = first(mine, "ip_class")
    if ip_class is None and intel is not None:
        ip_class = intel.classify(peer, asn).ip_class
    return {"ip": peer, "asn": asn,
            "asn_org": first(mine, "asn_org") or first(hops, "asn_org"),
            "country": first(mine, "geo_country") or first(hops, "geo_country"),
            "ip_class": ip_class,
            "basis": "ingest/ enrichment (ingest.geoip, or the dataset's own fields)"}


# --- an ASN --------------------------------------------------------------------
def asn_members(asn: int, src: Sources) -> list[str]:
    peers = set()
    if src.matrix is not None and "asn" in src.matrix:
        peers |= set(src.matrix.loc[src.matrix["asn"] == asn, "peer_ip"].astype(str))
    if src.transactions is not None and "asn" in src.transactions:
        peers |= set(src.transactions.loc[src.transactions["asn"] == asn,
                                          "src_ip"].astype(str))
    return sorted(p for p in peers if not is_onion(p))


def asn_profile(asn: int, src: Sources, cfg: dict | None = None) -> dict | None:
    """Aggregate over the ASN's member peers, with the per-peer breakdown."""
    cfg = cfg or config.load()
    members = []
    for peer in asn_members(asn, src):
        prof = peer_profile(peer, src, cfg)
        if prof is None:
            continue
        members.append({
            "peer": peer, "subject": prof["subject"],
            "originated": prof["originated"]["claimed"],
            "by_tier": prof["originated"]["by_tier"],
            "withheld": prof["withheld"]["count"], "relayed": prof["relayed"]["count"],
            "linked_clusters": [c["cluster_id"] for c in prof["linked_clusters"]],
            "timing_sufficient": prof["timing"]["sufficient"],
            "captures": sorted({v["capture_id"] or v["source"] for v in prof["vantage"]}),
            "simulated_only": prof["header"]["simulated_only"],
            "asn_org": prof["network"]["asn_org"], "country": prof["network"]["country"]})
    if not members:
        return None
    by_tier = Counter()
    for m in members:
        by_tier.update(m["by_tier"])
    clusters = sorted({c for m in members for c in m["linked_clusters"]})
    simulated_only = all(m["simulated_only"] for m in members)
    return {
        "subject": f"AS{asn}", "asn": asn, "peers": len(members),
        "header": {"simulated_only": simulated_only,
                   "statement": ("Built only from simulated or fixture data"
                                 if simulated_only else
                                 "Includes data from captures in a real format")},
        "totals": {"originated": sum(m["originated"] for m in members),
                   "by_tier": {t: by_tier.get(t, 0) for t in (validity.PASS,
                                                              validity.QUALIFIED,
                                                              validity.ANNOTATE)},
                   "withheld": sum(m["withheld"] for m in members),
                   "relayed": sum(m["relayed"] for m in members),
                   "linked_clusters": len(clusters)},
        "members": sorted(members, key=lambda m: (-m["originated"], -m["relayed"], m["peer"])),
        "caveat": (f"An aggregate over the {len(members)} peers seen with addresses in "
                   f"AS{asn}. An ASN is an address block announced by a network operator; "
                   "its peers are unrelated unless their own evidence says otherwise, so "
                   "read the per-peer rows, not only the totals."),
    }


# --- coverage, for the eval report -------------------------------------------
def coverage(src: Sources, cfg: dict | None = None) -> dict:
    """What share of observed peers get each part of a profile."""
    cfg = cfg or config.load()
    peers = set()
    if src.matrix is not None:
        peers |= set(src.matrix["peer_ip"].astype(str))
    if src.transactions is not None:
        peers |= set(src.transactions["src_ip"].astype(str))
    counts = Counter()
    for peer in sorted(peers):
        prof = peer_profile(peer, src, cfg)
        counts["peers"] += 1
        counts["onion identities"] += prof["kind"] == "onion_identity"
        counts["with an originated claim"] += prof["originated"]["claimed"] > 0
        counts["with a QUALIFIED claim"] += prof["originated"]["by_tier"]["QUALIFIED"] > 0
        counts["with an ANNOTATE claim"] += prof["originated"]["by_tier"]["ANNOTATE"] > 0
        counts["with a withheld top rank"] += prof["withheld"]["count"] > 0
        counts["relay only"] += (prof["originated"]["claimed"] == 0
                                 and prof["relayed"]["count"] > 0)
        counts["with a timing signature"] += prof["timing"]["sufficient"]
        counts["with a user agent"] += bool(prof["clients"]["user_agents"])
        counts["with service flags"] += bool(prof["clients"]["services"])
        counts["with a linked cluster"] += bool(prof["linked_clusters"])
        counts["with an ASN"] += (prof.get("network") or {}).get("asn") is not None
        counts["simulated/fixture only"] += prof["header"]["simulated_only"]
    return dict(counts)
