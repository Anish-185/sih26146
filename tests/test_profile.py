"""The reverse direction: peer (IP, onion identity, ASN) -> behavioural profile.

The relay-hop dataset is a real pipeline run over a small generated dataset,
because clusters and correlation leads need a real graph. The relay matrix is
written here, row by row, over txids of that dataset, so each case below — a
CoinJoin, an onion origin, a Dandelion-shaped arrival, an answer under the
cutoff, a peer too sparse for a timing signature — is stated rather than hoped
for. The model's probabilities are fixed per row; everything downstream of
them (`decide`, the validity verdicts, the abstention rule, the answer shapes)
is the production code.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import config
from analysis import validity
from engines.correlation import profile as P
from eval import origin as origin_eval
from graph.builder import iter_transactions
from graph.clustering import is_coinjoin
from origination.model import KEY, OriginationModel

CFG = config.load()
ONION = "abcdefghijklmnopqrstuvwxyz234567abcdefghijklmnopqrstuvwx.onion"
RELAYER = "203.0.113.60"
SPARSE = "203.0.113.61"
OBSERVER = "198.51.100.2"
ASN = 64500
#: Phrases a profile may never contain: each states or implies who someone is.
FORBIDDEN = ("owner of", "owned by", "belongs to", "belonging to", "operated by",
             "controlled by", "real-world", "real name", "identity of", "the person",
             "suspect")
#: Keys an onion-identity profile may never carry, at any depth.
NETWORK_KEYS = {"ip", "asn", "asn_org", "country", "geo_country", "ip_class", "src", "dst"}


class FixedModel(OriginationModel):
    """The production model with its probabilities replaced by the matrix's
    `p` column: decide(), validity and abstention are untouched."""

    def __init__(self, cutoff: float = 0.3):
        super().__init__(gbm=None, cutoff=cutoff)

    def score(self, matrix, calibrated=True):
        out = matrix[KEY + ["peer_ip"]].copy()
        out["abstain_reason"] = ["degenerate" if d else "scope_out" if s else ""
                                 for d, s in zip(matrix["degenerate"], matrix["scope_out"])]
        out["p_raw"] = out["p_calibrated"] = matrix["p"].where(out["abstain_reason"] == "")
        return out


def _row(txid, peer, capture, rank, n, delta, p, **extra):
    return {"txid": txid, "peer_ip": peer, "capture_id": capture,
            "observer_ip": OBSERVER, "capture_source": f"synthetic-fixture:{capture}",
            "direction": "inbound", "announce_ts": 1_790_244_000.0 + extra.pop("t") + delta,
            "announce_rank": rank, "candidate_count": n, "delta_vs_first_s": delta,
            "degenerate": n <= 1, "scope_out": False, "ip_class": "residential_or_unknown",
            "user_agent": None, "services": None, "asn": None, "asn_org": None,
            "geo_country": None, "unreadable_flows": None, "p": p, **extra}


def _coinjoin_rows(df: pd.DataFrame, peer: str) -> tuple[pd.DataFrame, str]:
    """A CoinJoin spending five existing wallets' coins, broadcast by `peer`."""
    wallets = list(dict.fromkeys(a for tx in iter_transactions(df) for a, _ in tx.inputs))[:5]
    txid = "c0" * 32
    base = df.iloc[0]
    rows = pd.DataFrame([{**base.to_dict(), "txid": txid, "src_ip": peer,
                          "dst_ip": "198.51.100.200", "timestamp": base["timestamp"],
                          "input_addresses": wallets, "input_amounts": [0.2] * 5,
                          "output_addresses": [f"bc1qmix{i}" for i in range(5)],
                          "output_amounts": [0.1] * 5, "fee": 0.0005}])
    return pd.concat([df, rows], ignore_index=True), txid


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from fusion.pipeline import run as fusion_run
    from generator.main import build_parser, generate
    from graph.builder import load
    from ingest.pipeline import run as ingest_run

    d = tmp_path_factory.mktemp("profile")
    raw = d / "raw"
    generate(build_parser().parse_args(["--n-actors", "120", "--n-transactions", "600",
                                        "--output", str(raw), "--seed", "23",
                                        "--formats", "csv"]))
    ingest_run(raw, d / "t.parquet", d / "q.parquet", "csv")
    base = load(d / "t.parquet")
    probe = P.build_sources(CFG, base, pd.DataFrame(), None)
    # The peer: the hop dataset's strongest correlation lead, so origination
    # and the lead can both link it to the same cluster.
    lead = probe.leads.iloc[0]
    peer, cluster = lead["ip"], lead["entity_id"]
    lead_txids = sorted({o.txid for o in probe.observations
                         if o.ip == peer and o.entity_id == cluster})
    others = [t for t in base["txid"].drop_duplicates() if t not in lead_txids
              and not is_coinjoin(probe.txs[t], CFG)][:30]
    df, mix = _coinjoin_rows(base, peer)
    df.to_parquet(d / "t.parquet", index=False)
    fusion_run(d / "t.parquet", raw / "ground_truth.json", d / "final.parquet",
               d / "final.json", CFG)

    rows, t = [], 0.0
    claimed = lead_txids + others[:20]
    for txid in claimed:                          # peer first and confident
        t += 60
        rows += [_row(txid, peer, "cap-a", 1, 3, 0.0, 0.9, t=t, user_agent="/Satoshi:27.0.0/",
                      services=1033, asn=ASN, geo_country="ZZ"),
                 _row(txid, RELAYER, "cap-a", 2, 3, 0.4, 0.05, t=t, asn=ASN),
                 _row(txid, "203.0.113.62", "cap-a", 3, 3, 0.8, 0.05, t=t)]
    t += 60                                        # under the cutoff: withheld
    rows += [_row(others[20], peer, "cap-a", 1, 2, 0.0, 0.1, t=t, asn=ASN),
             _row(others[20], RELAYER, "cap-a", 2, 2, 0.3, 0.05, t=t, asn=ASN)]
    t += 60                                        # the CoinJoin: QUALIFIED
    rows += [_row(mix, peer, "cap-a", 1, 2, 0.0, 0.9, t=t, asn=ASN),
             _row(mix, RELAYER, "cap-a", 2, 2, 0.2, 0.05, t=t, asn=ASN)]
    t += 60                                        # a stem's shape: ANNOTATE
    rows += [_row(others[21], peer, "cap-b", 1, 4, 0.0, 0.9, t=t, asn=ASN),
             *[_row(others[21], ip, "cap-b", k, 4, 10 + k / 10, 0.02, t=t)
               for k, ip in enumerate([RELAYER, SPARSE, "203.0.113.63"], start=2)]]
    for txid in others[22:24]:                     # an onion origin: QUALIFIED
        t += 60
        rows += [_row(txid, ONION, "cap-b", 1, 2, 0.0, 0.9, t=t),
                 _row(txid, SPARSE, "cap-b", 2, 2, 0.3, 0.05, t=t)]
    matrix = pd.DataFrame(rows)
    matrix.to_parquet(d / "relay.parquet", index=False)
    model = FixedModel()
    src = P.build_sources(CFG, load(d / "t.parquet"), matrix, model)
    return {"dir": d, "raw": raw, "src": src, "peer": peer, "cluster": cluster,
            "mix": mix, "withheld": others[20], "stem": others[21], "model": model,
            "matrix": matrix}


def _profiles(world):
    src = world["src"]
    peers = sorted(set(src.matrix["peer_ip"]) | set(src.transactions["src_ip"].astype(str)))
    return [P.peer_profile(p, src, CFG) for p in peers]


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


# --- what a profile holds ------------------------------------------------------
def test_originated_keeps_every_tier_and_never_counts_a_withheld_answer(world):
    prof = P.peer_profile(world["peer"], world["src"], CFG)
    tiers = {c["txid"]: c["tier"] for c in prof["originated"]["claims"]}
    assert tiers[world["mix"]] == validity.QUALIFIED
    assert tiers[world["stem"]] == validity.ANNOTATE
    assert prof["originated"]["by_tier"][validity.QUALIFIED] == 1
    assert prof["originated"]["by_tier"][validity.ANNOTATE] == 1
    mix = next(c for c in prof["originated"]["claims"] if c["txid"] == world["mix"])
    assert mix["answer"]["kind"] == "broadcasting_peer"
    assert "not attributable" in mix["answer"]["input_ownership"]
    assert "broadcasting peer" in mix["statement"]
    stem = next(c for c in prof["originated"]["claims"] if c["txid"] == world["stem"])
    assert validity.DANDELION_STEM in stem["statement"]
    for claim in prof["originated"]["claims"]:
        assert 0.0 <= claim["probability"] <= 1.0 and claim["calibration_basis"]
    assert world["withheld"] not in tiers
    assert prof["withheld"]["by_reason"] == {origin_eval.BELOW_CUTOFF: 1}


def test_relayed_is_a_count_and_a_sample(world):
    prof = P.peer_profile(RELAYER, world["src"], CFG)
    assert prof["originated"]["claimed"] == 0
    assert prof["relayed"]["count"] >= 20
    assert len(prof["relayed"]["sample"]) == CFG["engines"]["correlation"]["profile"][
        "relayed_sample"]


def test_client_history_and_vantage_say_which_capture_and_observer(world):
    prof = P.peer_profile(world["peer"], world["src"], CFG)
    agents = prof["clients"]["user_agents"]
    assert [a["value"] for a in agents] == ["/Satoshi:27.0.0/"]
    assert agents[0]["first_seen"] <= agents[0]["last_seen"]
    assert prof["clients"]["services"][0] == {**prof["clients"]["services"][0],
                                              "value": 1033, "hex": "0x409"}
    captures = {v["capture_id"]: v for v in prof["vantage"] if v["capture_id"]}
    assert set(captures) == {"cap-a", "cap-b"}
    assert all(v["observer"] == [OBSERVER] for v in captures.values())
    hop = next(v for v in prof["vantage"] if v["source"] == P.HOP_SOURCE)
    assert "receiving nodes" in hop["vantage"] and hop["observer"]
    assert prof["network"]["asn"] is not None and prof["network"]["country"]


def test_asn_aggregates_members_and_shows_each_one(world):
    prof = P.asn_profile(ASN, world["src"], CFG)
    peers = {m["peer"] for m in prof["members"]}
    assert {world["peer"], RELAYER} <= peers and ONION not in peers
    assert prof["peers"] == len(prof["members"])
    assert prof["totals"]["originated"] == sum(m["originated"] for m in prof["members"])
    assert prof["totals"]["relayed"] == sum(m["relayed"] for m in prof["members"])
    assert "per-peer" in prof["caveat"]


# --- honesty constraints ---------------------------------------------------------
def test_no_profile_states_or_implies_real_world_identity(world):
    everything = _profiles(world) + [P.asn_profile(ASN, world["src"], CFG)]
    for prof in everything:
        text = json.dumps(prof, default=str).lower()
        for phrase in FORBIDDEN:
            assert phrase not in text, (phrase, prof["subject"])
        assert prof["subject"].startswith(("peer ", "onion identity ", "AS"))
        for c in prof.get("linked_clusters", []):
            assert c["cluster"] == f"cluster {c['cluster_id']}"
            assert c["statement"].startswith("peer ")


@pytest.mark.parametrize("enforce", [True, False])
def test_a_coinjoin_never_links_a_cluster(world, enforce):
    """Even with the validity layer off — so the answer is not qualified — the
    CoinJoin's structure still keeps it out of every cluster link."""
    cfg = {**CFG, "validity": {**CFG["validity"], "enforce": enforce}}
    src = P.build_sources(cfg, world["src"].transactions, world["matrix"], world["model"],
                          world["src"].features)
    prof = P.peer_profile(world["peer"], src, cfg)
    for link in prof["linked_clusters"]:
        assert world["mix"] not in {e["txid"] for e in link["evidence"]}
    assert {e["txid"]: e["reason"] for e in prof["excluded_links"]}[world["mix"]] == (
        validity.COINJOIN)


def test_a_coinjoin_the_verdict_missed_still_links_no_cluster(world):
    """Defence in depth: an answer whose verdict lost the COINJOIN reason (a
    stored output, a detector change) is still kept out by the structure."""
    src = world["src"]
    answers = src.answers.copy()
    mix = answers["txid"] == world["mix"]
    answers.loc[mix, "validity"] = validity.PASS
    answers.loc[mix, "validity_tier"] = validity.PASS
    answers["validity_reasons"] = [[] if m else r for m, r in zip(mix, answers["validity_reasons"])]
    answers["answer"] = [{"kind": "ip_attribution", "ip": world["peer"]} if m else a
                         for m, a in zip(mix, answers["answer"])]
    missed = P.Sources(**{**src.__dict__, "answers": answers, "_index": {}})
    prof = P.peer_profile(world["peer"], missed, CFG)
    assert world["mix"] in {c["txid"] for c in prof["originated"]["claims"]}
    for link in prof["linked_clusters"]:
        assert world["mix"] not in {e["txid"] for e in link["evidence"]}


def test_every_linked_cluster_traces_back_to_raw_rows(world):
    src = world["src"]
    hops, matrix = src.transactions, src.matrix
    linked = 0
    for prof in _profiles(world):
        for link in prof["linked_clusters"]:
            linked += 1
            assert link["evidence"] and link["evidence_total"] >= len(link["evidence"])
            assert link["basis"] in ("origination", "correlation lead", "both")
            assert set(link["by_basis"]) == ({"origination", "correlation lead"}
                                             if link["basis"] == "both" else {link["basis"]})
            for ev in link["evidence"]:
                assert ev["rows"], "a link with no row behind it"
                for row in ev["rows"]:
                    if row["source"] == P.HOP_SOURCE:
                        assert hops.loc[row["row"], "txid"] == ev["txid"]
                        assert str(hops.loc[row["row"], "src_ip"]) == row["src"]
                    else:
                        hit = matrix[(matrix["capture_id"] == row["capture_id"])
                                     & (matrix["txid"] == ev["txid"])
                                     & (matrix["peer_ip"] == row["peer"])]
                        assert len(hit) == 1
                tx_inputs = {a for a, _ in src.txs[ev["txid"]].inputs}
                assert ev["inputs"] and set(ev["inputs"]) <= tx_inputs
                assert {src.features.entity_of(a) for a in ev["inputs"]} == {link["cluster_id"]}
    assert linked, "the fixture must exercise at least one link"


def test_origination_and_a_lead_on_the_same_cluster_is_basis_both(world):
    prof = P.peer_profile(world["peer"], world["src"], CFG)
    link = next(c for c in prof["linked_clusters"] if c["cluster_id"] == world["cluster"])
    assert link["basis"] == "both"
    assert link["confidence"] == max(b["confidence"] for b in link["by_basis"].values())


def test_too_few_observations_get_a_sentence_not_a_signature(world):
    need = CFG["engines"]["correlation"]["profile"]["min_timing_observations"]
    sparse = P.peer_profile(SPARSE, world["src"], CFG)["timing"]
    assert sparse["announcements"] < need and not sparse["sufficient"]
    assert set(sparse) == {"sufficient", "announcements", "threshold", "statement"}
    assert str(need) in sparse["statement"]
    busy = P.peer_profile(world["peer"], world["src"], CFG)["timing"]
    assert busy["sufficient"] and len(busy["active_hours_utc"]) == 24
    assert sum(busy["active_hours_utc"]) == busy["announcements"]


def test_simulated_or_fixture_only_profiles_say_so(world):
    prof = P.peer_profile(world["peer"], world["src"], CFG)
    assert prof["header"]["simulated_only"]
    assert "not the Bitcoin network" in prof["header"]["statement"]
    real = world["matrix"].assign(capture_source="debug.log:node1.log")
    src = P.build_sources(CFG, None, real, world["model"])
    assert not P.peer_profile(RELAYER, src, CFG)["header"]["simulated_only"]


def test_an_onion_identity_never_carries_an_ip_asn_or_country(world):
    prof = P.peer_profile(ONION, world["src"], CFG)
    assert prof["kind"] == "onion_identity" and prof["subject"].startswith("onion identity")
    assert not NETWORK_KEYS & set(_keys(prof)), NETWORK_KEYS & set(_keys(prof))
    assert all(c["tier"] == validity.QUALIFIED and c["answer"]["kind"] == "onion_identity"
               for c in prof["originated"]["claims"])
    assert prof["originated"]["claimed"] == 2


# --- the abstention codes --------------------------------------------------------
def test_abstention_reasons_are_exactly_the_flagged_rows(world):
    frame = world["src"].answers
    for cutoff in (0.0, 0.3, 0.95):
        reasons = origin_eval.abstention_reasons(frame, CFG, cutoff)
        assert (reasons.notna() == origin_eval.flagged_at(frame, CFG, cutoff)).all()


# --- the API -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(world):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api.app as app_module
    d = world["dir"]
    world["model"].save(d / "model.pkl")
    app_module.configure(d / "t.parquet", world["raw"], d / "final.json",
                         d / "feedback.parquet", d / "relay.parquet", d / "model.pkl")
    yield TestClient(app_module.app)
    app_module.configure()


def test_round_trip_every_claimed_origin_shows_on_its_txid_page(world, client):
    """Reverse -> forward: each transaction a profile says peer X originated
    names peer X on that transaction's own page, in the same capture, with the
    same tier and the same answer."""
    checked = 0
    for peer in (world["peer"], ONION):
        prof = client.get(f"/peers/{peer}/profile").json()
        for claim in prof["originated"]["claims"]:
            page = client.get(f"/transactions/{claim['txid']}/origination").json()
            there = next(c for c in page["captures"] if c["capture_id"] == claim["capture_id"])
            assert there["answered"] and there["named_peer"] == peer
            assert there["answer"] == claim["answer"]
            assert there["validity"]["tier"] == claim["tier"]
            checked += 1
    assert checked >= 20


def test_forward_to_reverse_every_answered_origin_is_in_its_peers_profile(world, client):
    for row in world["src"].answers[world["src"].answers["answered"]].itertuples():
        prof = client.get(f"/peers/{row.estimated_origin_ip}/profile").json()
        assert (row.capture_id, row.txid) in {(c["capture_id"], c["txid"])
                                              for c in prof["originated"]["claims"]}


def test_every_profile_lookup_is_in_the_custody_ledger(world, client):
    import custody
    before = len(custody.read())
    assert client.get(f"/peers/{world['peer']}/profile").status_code == 200
    assert client.get("/peers/192.0.2.254/profile").status_code == 404
    assert client.get(f"/asns/AS{ASN}/profile").status_code == 200
    entries = custody.read()[before:]
    assert [(e["action"], e["detail"]["subject"], e["detail"]["found"]) for e in entries] == [
        ("lookup.peer_profile", world["peer"], True),
        ("lookup.peer_profile", "192.0.2.254", False),
        ("lookup.asn_profile", f"AS{ASN}", True)]


def test_asn_endpoint_rejects_a_non_number(client):
    assert client.get("/asns/cloudflare/profile").status_code == 422


def test_the_eval_abstention_breakdown_sums_to_the_abstentions(world):
    from analysis.evaluate import ABSTENTION_CODES, POLICIES, abstention_breakdown
    frame = world["src"].answers.assign(correct=True)
    cuts = {(p, m): 0.3 for p in POLICIES for m in origin_eval.METRICS}
    rows = {r["policy"]: r for r in abstention_breakdown(frame, cuts, CFG, "fixture", "t")}
    for row in rows.values():
        assert sum(row[code] for code in ABSTENTION_CODES) == row["abstained"]
        assert row["abstention rate"] == round(row["abstained"] / row["n"], 3)
    assert rows["tiered"][origin_eval.BELOW_CUTOFF] == 1
    assert rows["tiered"][validity.COINJOIN] == 0          # QUALIFIED answers
    assert rows["binary"][validity.COINJOIN] == 1          # P6's gate withholds them
    assert rows["off"]["abstained"] == rows["off"][origin_eval.BELOW_CUTOFF]


def test_the_demo_capture_shares_the_served_datasets_txids(world, tmp_path):
    """p2p.demo_capture folds the relay-hop log into one capture whose txids are
    the served ones, readable by the ordinary capture reader."""
    from p2p import capture_reader, demo_capture
    hops = world["src"].transactions
    out = demo_capture.write(hops, tmp_path)
    events = capture_reader.read_capture(out["path"])
    assert {e.txid for e in events} == set(hops["txid"])
    assert all(e.capture_source == "sim:demo-hoplog" for e in events)
    assert P.capture_provenance(events[0].capture_source) == "simulated"
    per_tx = {}
    for e in events:
        per_tx.setdefault(e.txid, set()).add(e.peer_ip)
    senders = hops.groupby("txid")["src_ip"].nunique()
    assert all(len(per_tx[t]) == n for t, n in senders.items())


def test_every_actor_link_traces_to_raw_evidence_and_skips_the_coinjoin(world):
    """docs/ACTORS.md: each link carries the raw rows it rests on, and the
    CoinJoin the peer is QUALIFIED origin of joins no participant's cluster."""
    from fusion import actors as A
    links = A.peer_links(world["src"], CFG)
    assert links
    for lk in links:
        assert lk["evidence_total"] >= 1 and lk["evidence"]
        assert all(e["rows"] for e in lk["evidence"]), (lk["peer"], lk["cluster_id"])
        assert all(e["txid"] != world["mix"] for e in lk["evidence"]
                   if e["basis"] == "origination")
    mix_clusters = {world["src"].features.entity_of(a)
                    for a, _ in world["src"].txs[world["mix"]].inputs}
    joined_via_mix = [lk for lk in links if lk["cluster_id"] in mix_clusters and A.may_join(lk)
                      and all(e["txid"] == world["mix"] for e in lk["evidence"])]
    assert not joined_via_mix
    assert any(lk["peer_kind"] == "onion identity" and lk["ip_class"] is None for lk in links) \
        or not any(P.is_onion(lk["peer"]) for lk in links)
