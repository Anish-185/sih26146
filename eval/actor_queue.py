"""The actor queue against the entity queue, on generator ground truth: triage.

Both queues come from one dataset, one fusion bundle and one fitted stacker, so
the only difference is the unit: an entity (address cluster) or an actor
(clusters joined through peer identities, fusion/actors.py). Ground truth is
`eval.actors.actors_of`: the illicit operations pre-registered in
docs/detection_unit_protocol.md.

A queue item *finds* an operation when it holds any of the operation's
wallets. A wallet's true controller, for purity, is its operation if it has
one, else its generator actor.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from engines.correlation.profile import build_sources
from features import relay
from fusion import actors as A
from fusion.pipeline import build_alerts, collect_signals
from ingest.ip_intel import load_intel
from p2p import demo_capture

from .actors import actors_of
from .datasets import Dataset

KS = (10, 25, 50)


def dataset_actors(dataset: Dataset, stacker, cfg: dict) -> dict:
    """The fusion bundle, the entity alerts and the actors of one dataset.
    Its relay matrix is built the way the served one is (p2p.demo_capture,
    then features.relay), in a scratch directory."""
    cfg = json.loads(json.dumps(cfg))
    cfg["ingest"]["input_dir"] = str(dataset.raw)       # this dataset's node intel
    df = dataset.frame()
    bundle = collect_signals(df, cfg, dataset.raw)
    alerts = build_alerts(bundle, stacker, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        capture = Path(tmp) / "demo_capture"
        demo_capture.write(df, capture)
        matrix, _, _ = relay.build(capture, [demo_capture.COLLECTOR], cfg)
    src = build_sources(cfg, transactions=df, matrix=matrix, features=bundle["features"],
                        intel=load_intel(None, dataset.raw, cfg))
    links = A.peer_links(src, cfg)
    actors = A.build(bundle["signals"], stacker, links, cfg, set(alerts["entity_id"]))
    return {"bundle": bundle, "alerts": alerts, "actors": actors, "links": links}


def _wallets(entity: str, clusters: dict) -> set[str]:
    return set(clusters.get(entity, {entity}))


def triage(dataset: Dataset, built: dict) -> dict:
    gt = dataset.ground_truth()
    operations = actors_of(dataset)
    clusters = built["bundle"]["features"].clustering.clusters
    op_of = {w: op.actor_id for op in operations for w in op.wallets}
    controller = lambda w: op_of.get(w) or gt["wallets"].get(w, w)
    present = {op.actor_id for op in operations
               if any(built["bundle"]["features"].entity_of(w) for w in op.wallets)}

    entity_items = [_wallets(e, clusters) for e in built["alerts"]["entity_id"]]
    queue = A.queue(built["actors"])
    actor_items = [set().union(*(_wallets(m, clusters) for m in members))
                   for members in queue["members"]]

    def found(items):
        return [{op_of[w] for w in wallets if w in op_of} for wallets in items]

    def rows(name, items):
        hits = found(items)
        out = {"queue": name, "items": len(items)}
        for k in KS:
            top = hits[:k]
            seen = set().union(*top) if top else set()
            out[f"precision@{k}"] = round(sum(bool(h) for h in top) / len(top), 3) if top else None
            out[f"recall@{k}"] = round(len(seen) / len(present), 3) if present else None
        # A generator dataset holds only a handful of illicit operations, so
        # "items reviewed before finding N" is asked for 1, 3 and all of them.
        for n in sorted({1, min(3, len(present)), len(present)} - {0}):
            reviewed = None
            seen = set()
            for i, h in enumerate(hits, 1):
                seen |= h
                if len(seen) >= n:
                    reviewed = i
                    break
            label = "all" if n == len(present) else str(n)
            out[f"reviewed to find {label}"] = reviewed if reviewed is not None else "not reached"
        return out

    table = [rows("entity queue (before)", entity_items), rows("actor queue", actor_items)]

    def purity(wallets):
        c = Counter(controller(w) for w in wallets)
        return c.most_common(1)[0][1] / sum(c.values())

    multi = queue[queue["members"].map(len) > 1]
    majority = lambda m: Counter(controller(w) for w in _wallets(m, clusters)).most_common(1)[0][0]
    wrong = [len({majority(m) for m in members}) > 1 for members in multi["members"]]
    all_multi = built["actors"][built["actors"]["members"].map(len) > 1]
    wrong_all = [len({majority(m) for m in members}) > 1 for members in all_multi["members"]]
    quality = {
        "entity alerts": len(entity_items), "actor alerts": len(actor_items),
        "alert count reduction": round(1 - len(actor_items) / len(entity_items), 3)
        if entity_items else None,
        "illicit operations present": len(present),
        "actor purity (alerted, mean)": round(sum(map(purity, actor_items)) / len(actor_items), 3)
        if actor_items else None,
        "entity purity (alerted, mean)": round(sum(map(purity, entity_items)) / len(entity_items), 3)
        if entity_items else None,
        "alerted actors joining 2+ clusters": len(multi),
        "wrong-merge rate (alerted multi-cluster actors)": round(sum(wrong) / len(wrong), 3)
        if wrong else None,
        "actors joining 2+ clusters (all)": len(all_multi),
        "wrong-merge rate (all multi-cluster actors)": round(sum(wrong_all) / len(wrong_all), 3)
        if wrong_all else None,
        "links": len(built["links"]),
        "joining links": sum(A.may_join(lk) for lk in built["links"]),
    }
    return {"table": pd.DataFrame(table), "quality": quality}


def evaluate(datasets: dict[str, Dataset], stackers: dict, cfg: dict) -> dict:
    out = {}
    for name, dataset in datasets.items():
        built = dataset_actors(dataset, stackers[name], cfg)
        out[name] = triage(dataset, built)
    return out
