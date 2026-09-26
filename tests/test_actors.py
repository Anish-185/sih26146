"""fusion/actors.py: joining, scoring through the one stacker, and naming."""

from __future__ import annotations

import inspect

import pandas as pd

import config
from fusion import actors as A
from fusion.stacker import SIGNALS, Stacker

CFG = config.load()
STACKER = Stacker(fallback_weights={s: 1.0 for s in SIGNALS})


def signals(**by_entity):
    rows = [{"entity_id": e, **dict.fromkeys(SIGNALS, 0.0), **v} for e, v in by_entity.items()]
    return pd.DataFrame(rows)


def link(peer, cluster, basis="both", confidence=0.6, ip_class="residential_or_unknown",
         reasons=(), kind="IP"):
    return {"peer": peer, "peer_kind": kind, "ip_class": ip_class, "cluster_id": cluster,
            "basis": basis, "confidence": confidence, "by_basis": {}, "origin_tiers": ["PASS"],
            "evidence": [{"basis": "origination", "txid": f"t-{peer}-{cluster}",
                          "probability": confidence, "tier": "PASS", "reasons": list(reasons),
                          "rows": [{"row": 1}]}],
            "evidence_total": 1, "evidence_chain": "raw rows -> transaction -> inputs -> cluster"}


def actor_of(frame, entity):
    return frame[frame["members"].map(lambda m: entity in m)].iloc[0]


def test_a_corroborated_residential_link_joins_two_clusters():
    s = signals(c1={"rule_score": 0.9}, c2={"taint_score": 0.5}, c3={})
    out = A.build(s, STACKER, [link("10.0.0.1", "c1", confidence=0.8),
                               link("10.0.0.1", "c2", confidence=0.4)], CFG)
    joined = actor_of(out, "c1")
    assert joined["members"] == ["c1", "c2"] and joined["peers"] == ["10.0.0.1"]
    assert joined["membership_confidence"] == 0.4          # the weakest link
    assert actor_of(out, "c3")["members"] == ["c3"]        # zero peers is an actor too


def test_shared_infrastructure_and_lead_only_links_attach_but_never_join():
    s = signals(c1={"rule_score": 0.9}, c2={"rule_score": 0.1})
    for lk in ([link("5.5.5.5", "c1", ip_class="hosting_vpn"), link("5.5.5.5", "c2", ip_class="hosting_vpn")],
               [link("10.0.0.2", "c1", basis="correlation lead"),
                link("10.0.0.2", "c2", basis="correlation lead", confidence=0.05)]):
        out = A.build(s, STACKER, lk, CFG)
        assert len(out) == 2, "clusters stay separate actors"
        weak = actor_of(out, "c2")
        assert weak["peers"] and not weak["links"][0]["joins"]
        assert weak["membership_confidence"] == lk[1]["confidence"]   # lowered, not dropped


def test_a_coinjoin_qualified_origination_never_joins_participants():
    s = signals(c1={"rule_score": 0.9}, c2={})
    mix = [link("10.0.0.3", "c1", reasons=["COINJOIN"]), link("10.0.0.3", "c2", reasons=["COINJOIN"])]
    assert not any(A.may_join(lk) for lk in mix)
    out = A.build(s, STACKER, mix, CFG)
    assert all(len(m) == 1 for m in out["members"])
    assert all(not r for r in out["links"]), "a COINJOIN-only link is not a membership link"


def test_the_actor_is_scored_by_the_same_stacker_on_weighted_signals():
    s = signals(c1={"rule_score": 0.9}, c2={"taint_score": 0.8})
    out = A.build(s, STACKER, [link("10.0.0.1", "c1", confidence=0.9),
                               link("10.0.0.1", "c2", confidence=0.5)], CFG)
    joined = actor_of(out, "c1")
    expected = dict.fromkeys(SIGNALS, 0.0) | {"rule_score": 0.9, "taint_score": 0.8 * 0.5}
    assert joined["signals"] == {k: round(v, 4) for k, v in expected.items()}
    assert abs(joined["risk_score"] - STACKER.score(pd.DataFrame([expected]))[0]) < 1e-12
    alone = A.build(s, STACKER, [], CFG)
    assert abs(actor_of(alone, "c1")["risk_score"] - STACKER.score(s)[0]) < 1e-12


def test_actors_are_handles_not_people():
    out = A.build(signals(c1={}, c2={}), STACKER, [], CFG)
    assert all(n == f"actor {i}" and i.startswith("A-") for n, i in zip(out["name"], out["actor_id"]))
    assert "does not name or imply a person or an organisation" in A.STATEMENT
    for word in ("owner", "person ", "suspect", "operated by", "belongs to"):
        assert word not in out["name"].str.cat().lower()


def test_fingerprints_contribute_nothing_to_membership():
    source = inspect.getsource(A)
    assert "import fingerprint" not in source and "fingerprint." not in source
