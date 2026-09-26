"""Actors: an address cluster joined to the peer identities the evidence ties it to.

An alert today is an entity: one address cluster from graph/ (or a lone
wallet). The network side already exists beside it: engines/correlation's
profiles link a peer (IP or onion identity) to clusters, by origination claim
or by correlation lead, each link with a basis, a confidence and the raw rows
behind it. An actor is what those links join:

    actor = one or more clusters + zero or more peers, connected by links

Membership comes only from those links. Fingerprints contribute nothing, and a
CoinJoin's origination never joins its broadcaster to the participants'
clusters: `profile._linked_clusters` excludes it before a link exists, and
`build` refuses any link whose origination evidence is COINJOIN-qualified.

JOINING, FIXED BEFORE ANY EVALUATION. Every link is kept and shown, but only a
link that `may_join` can put two clusters in one actor: the peer is a
residential/unknown IP or an onion identity (a hosting/VPN range, Tor exit or
public relay serves many unrelated users by definition, as the validity layer
already treats it), and an origination claim stands behind the link (basis
"origination" or "both"), not a correlation lead alone. Any other link attaches
its peer to the actor of the cluster it names, lowering that actor's
membership confidence, and merges nothing.

SCORING — ONE PATH. An actor is scored by the same fitted stacker as an entity,
on the same signals. Each signal is the largest member value, weighted by how
sure the join to that member is: the anchor (the member the stacker scores
highest) counts in full, every other member by the weakest link on the
strongest path from the anchor. So an uncertain join lowers what a member can
add, and lowers the actor's membership confidence (its weakest joining link),
rather than being dropped.

Actors are named "actor A-17". A name is a handle for a group of addresses
and network identities that the evidence links; it never implies a person or
an organisation.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

import config
from analysis.validity import COINJOIN
from engines.correlation import profile as correlation_profile

#: What the actor's membership confidence is, in words; the parts are not all calibrated.
CONFIDENCE_BASIS = ("the weakest link joining the actor: origination links are "
                    "1 - exp(-sum of calibrated probabilities / k); correlation leads are "
                    "an evidence-weighted count, not calibrated (fusion.pipeline)")
JOINING_CLASSES = ("residential_or_unknown",)
JOINING_BASES = ("origination", "both")
STATEMENT = ("an actor is a group of addresses and network identities linked by evidence; "
             "it does not name or imply a person or an organisation")


def peers_of(src) -> list[str]:
    """Every peer a link could come from: named origins and correlation-lead IPs."""
    peers = set()
    if src.answers is not None and len(src.answers):
        peers |= set(src.answers.loc[src.answers["answered"], "estimated_origin_ip"].astype(str))
    if src.leads is not None and len(src.leads):
        peers |= set(src.leads["ip"].astype(str))
    return sorted(peers)


def peer_links(src, cfg: dict) -> list[dict]:
    """(peer, cluster) links with basis, confidence, the validity tiers of any
    origin claims behind them, and their raw evidence — from the P7 profiles."""
    out = []
    for peer in peers_of(src):
        prof = correlation_profile.peer_profile(peer, src, cfg)
        if prof is None:
            continue
        onion = correlation_profile.is_onion(peer)
        ip_class = None if onion or src.intel is None else src.intel.classify(peer).ip_class
        for link in prof["linked_clusters"]:
            evidence = link["evidence"]
            tiers = sorted({e["tier"] for e in evidence if e["basis"] == "origination"})
            out.append({
                "peer": peer, "peer_kind": "onion identity" if onion else "IP",
                "ip_class": ip_class, "cluster_id": link["cluster_id"], "basis": link["basis"],
                "confidence": float(link["confidence"]), "by_basis": link["by_basis"],
                "origin_tiers": tiers, "evidence": evidence,
                "evidence_total": link["evidence_total"],
                "evidence_chain": link["evidence_chain"]})
    return out


def may_join(link: dict) -> bool:
    """Whether this link may put its cluster in the same actor as the peer's
    other clusters (see JOINING in the module docstring)."""
    return ((link["peer_kind"] == "onion identity" or link["ip_class"] in JOINING_CLASSES)
            and link["basis"] in JOINING_BASES and not _joins_through_coinjoin(link))


def _joins_through_coinjoin(link: dict) -> bool:
    return any(e["basis"] == "origination" and COINJOIN in e.get("reasons", [])
               for e in link["evidence"])


def build(signals: pd.DataFrame, stacker, links: list[dict], cfg: dict | None = None,
          alert_ids: set[str] | frozenset = frozenset()) -> pd.DataFrame:
    """One row per actor, highest risk first.

    `signals` is the fusion bundle's per-entity signal frame (every entity, not
    only the alerted ones); `stacker` the fitted fusion stacker; `links` from
    `peer_links`; `alert_ids` the entity alerts that exist, for drill-down.
    """
    cfg = cfg or config.load()
    threshold = cfg["fusion"]["alert_threshold"]
    names = list(stacker.signals)
    frame = signals.set_index("entity_id")
    entity_risk = pd.Series(stacker.score(frame.reset_index()), index=frame.index)
    links = [lk for lk in links if lk["cluster_id"] in frame.index
             and not _joins_through_coinjoin(lk)]

    # Kruskal on descending confidence: the tree it builds holds, between any
    # two members, the path whose weakest link is strongest.
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = defaultdict(list)
    peer_node = lambda p: f"peer:{p}"
    joining = [lk for lk in links if may_join(lk)]
    for lk in sorted(joining, key=lambda k: (-k["confidence"], k["peer"], k["cluster_id"])):
        a, b = peer_node(lk["peer"]), lk["cluster_id"]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            tree[a].append((b, lk["confidence"]))
            tree[b].append((a, lk["confidence"]))
    components = defaultdict(set)
    for entity in frame.index:
        components[find(entity)].add(entity)
    for lk in joining:
        components[find(lk["cluster_id"])].add(peer_node(lk["peer"]))

    by_cluster = defaultdict(list)
    for lk in links:
        by_cluster[lk["cluster_id"]].append(lk)

    rows = []
    for nodes in components.values():
        clusters = sorted(n for n in nodes if not n.startswith("peer:"))
        member_links = [lk for c in clusters for lk in by_cluster.get(c, [])]
        peers = sorted({lk["peer"] for lk in member_links})
        anchor = max(clusters, key=lambda c: (entity_risk[c], c))
        weight = _widest_paths(anchor, tree)
        member_signals = frame.loc[clusters, names].astype(float).fillna(0.0)
        weighted = member_signals.mul([weight.get(c, 0.0) for c in clusters], axis=0)
        actor_signals = weighted.max().to_frame().T
        score = float(stacker.score(actor_signals)[0])
        rows.append({
            "members": clusters, "peers": peers, "anchor": anchor, "risk_score": score,
            # Every link counts here, joining or not: an uncertain peer
            # lowers how sure the actor is, even when it merges nothing.
            "membership_confidence": min((lk["confidence"] for lk in member_links),
                                         default=1.0),
            "signals": {s: round(float(actor_signals[s].iloc[0]), 4) for s in names},
            "member_detail": [{"entity_id": c, "entity_risk": round(float(entity_risk[c]), 4),
                               "weight": round(weight.get(c, 0.0), 4),
                               "alerted": c in alert_ids} for c in clusters],
            "links": [{k: v for k, v in lk.items() if k != "evidence"}
                      | {"evidence": lk["evidence"][:5], "joins": may_join(lk)}
                      for lk in member_links],
            "origin_tiers": sorted({t for lk in member_links for t in lk["origin_tiers"]}),
            "alerted": score >= threshold,
        })
    out = pd.DataFrame(rows)
    # Stable names: numbered by smallest member, so re-sorting never renames an actor.
    out = out.sort_values("members", key=lambda s: s.map(lambda m: m[0])).reset_index(drop=True)
    out["actor_id"] = [f"A-{i}" for i in range(1, len(out) + 1)]
    out["name"] = "actor " + out["actor_id"]
    out["statement"] = STATEMENT
    out["confidence_basis"] = CONFIDENCE_BASIS
    out = out.sort_values(["risk_score", "membership_confidence", "actor_id"],
                          ascending=[False, False, True], kind="stable").reset_index(drop=True)
    return out


def _widest_paths(anchor: str, tree: dict) -> dict[str, float]:
    """Node -> the weakest link on the tree path from `anchor` (1.0 at the anchor)."""
    best, stack = {anchor: 1.0}, [anchor]
    while stack:
        node = stack.pop()
        for nxt, conf in tree.get(node, ()):
            if nxt not in best:
                best[nxt] = min(best[node], conf)
                stack.append(nxt)
    return best


def queue(actors: pd.DataFrame) -> pd.DataFrame:
    """The actor queue: alerted actors, highest risk first."""
    return actors[actors["alerted"]].reset_index(drop=True)


def to_json(actors: pd.DataFrame) -> str:
    return json.dumps(json.loads(actors.to_json(orient="records")), indent=1)
