"""One command, one canonical results file.

    python -m eval.report

Everything in eval/results.md comes from here, with the seed and sizes fixed in
config.yaml's `eval:` block. Ad-hoc runs are how this repo ended up quoting two
different ceiling figures for the same statistic; there is now one source.

The unit of detection is the actor, pre-registered in
docs/detection_unit_protocol.md before any of these numbers existed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

from . import (clustering_eval, correlation_eval, fusion_eval, origin,
               redteam_batch, saturation, saturation_diagnosis, zero_attack)
from .ground_truth import score as ground_truth
from .datasets import build


def md_table(df: pd.DataFrame, floats: int = 3) -> str:
    if df.empty:
        return "_(no rows)_\n"
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(
                lambda v: "n/a" if pd.isna(v) else f"{v:.{floats}f}")
    formatted = formatted.fillna("n/a")
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    rule = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in formatted.itertuples(index=False))
    return f"{header}\n{rule}\n{body}\n"


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def choose_default_estimator(table: pd.DataFrame, cfg: dict) -> tuple[str, str]:
    """Apply docs/origin_eval_protocol.md, without re-reading the rule."""
    rate = cfg["eval"]["default_rate"]
    at_rate = table[table["rate"] == rate].sort_values("top1", ascending=False)
    best = at_rate.iloc[0]
    runner = at_rate.iloc[1] if len(at_rate) > 1 else None
    tied = runner is not None and abs(best["top1"] - runner["top1"]) < 0.01
    if tied and "first_timestamp" in set(at_rate.head(2)["estimator"]):
        return "first_timestamp", (
            f"tie at rate {rate} ({best['estimator']} {best['top1']:.3f} vs "
            f"{runner['estimator']} {runner['top1']:.3f}, under 1pp) — "
            "protocol awards ties to first_timestamp")
    return best["estimator"], (
        f"best top-1 at rate {rate}: {best['top1']:.3f}"
        + (f" vs {runner['estimator']} {runner['top1']:.3f}" if runner is not None else ""))


def gnn_provenance(cfg: dict) -> dict:
    """Which dataset the GNN was trained on, and whether that is one of ours.

    The demo dataset in `data/raw` uses the same seed and size as the canonical
    evaluation set. A GNN trained the obvious way is therefore trained on
    exactly what this report scores it against, and would post a near-perfect
    `gnn_score` that means nothing. Older models record no provenance at all,
    which is treated as unknown rather than as safe.
    """
    path = Path(cfg["models"]["gnn"])
    e = cfg["eval"]
    reserved = {e["seed"], e["seed_b"]}
    if not path.exists():
        return {"status": "absent", "usable": False,
                "detail": f"no model at {path} — `gnn_score` is 0 everywhere below"}
    try:
        import torch
        trained_on = torch.load(path, map_location="cpu", weights_only=False).get("trained_on")
    except Exception as exc:
        return {"status": "unreadable", "usable": False,
                "detail": f"{path} could not be read ({type(exc).__name__})"}
    if not trained_on:
        return {"status": "unknown provenance", "usable": False,
                "detail": (f"{path} records no training dataset. It predates the check, "
                           "so it cannot be shown to be held out from these seeds")}
    seed = trained_on.get("seed")
    if seed in reserved:
        return {"status": "LEAKED", "usable": False, "seed": seed,
                "detail": (f"the model was trained on seed {seed}, which is an evaluation "
                           f"seed. Retrain on `eval.gnn_train_seed` ({e['gnn_train_seed']}) "
                           "before any GNN number here means anything")}
    return {"status": "held out", "usable": True, "seed": seed,
            "detail": f"trained on seed {seed}; evaluation seeds are {sorted(reserved)}"}


def fusion_section(add, result: dict, label: str, dataset) -> None:
    add(f"\n### {label} — {dataset.describe()}\n")
    add(f"{result['entities']} entities, {result['actors']} illicit actors, "
        f"{result['rule_alerts']} rule alerts, {result['alerted_entities']} alerted "
        f"entities, {result['watchlist_seeds']} watchlist seed entities.\n")

    add("\n**Actor-level detection** (label: entity holds a wallet of a "
        "ground-truth illicit operation):\n\n")
    add(md_table(pd.DataFrame([result["cases"]])))
    add("\n**Per typology:**\n\n")
    add(md_table(result["per_typology_cases"]))
    add("\n`trace_coverage@N` is scored only on detected actors that still have "
        "un-alerted\nwallets left to reach; `traceable@N` is how many actors that was. "
        "`n/a` means\nevery wallet of every detected actor was already alerted.\n")

    for name, note in (("actor", "an entity holding an illicit actor's wallet"),
                       ("broad", "every wallet the money passed through — the old label")):
        block = result.get(name)
        if not block:
            continue
        caveat = (" _(in-sample: too few positives for a chronological hold-out)_"
                  if block.get("auc_in_sample") else "")
        add(f"\n**Stacker, label `{name}`** ({note}) — {block['positives']} positive "
            f"entities, AUC **{block['auc']}**{caveat}\n\n")
        rows = [{"signal": s, "alone": block["signal_auc"].get(s),
                 "stack without it": block["ablation"].get(s, {}).get("stack_without_it"),
                 "weight": block["coefficients"].get(s)}
                for s in block["ablation"]]
        add(md_table(pd.DataFrame(rows), floats=4))

    taint = result.get("taint", {})
    if taint:
        add("\n**Taint, scored only on entities that were neither watchlist seeds nor "
            "rule-flagged:**\n\n")
        add(md_table(pd.DataFrame([taint]), floats=3))
    add("\n**Wallet-level coverage per typology (secondary):**\n\n")
    add(md_table(result["per_typology_wallets"]))


def summary_table(fusion: dict, origin_rows: dict, cfg: dict, extra: dict) -> pd.DataFrame:
    """The numbers that would go on a slide, each with where it came from."""
    e = cfg["eval"]
    std, shift = fusion["standard"], fusion["shifted"]
    actor_label = "actor (entity holds an illicit operation's wallet)"
    rows = [
        ("case_detection_rate", fmt(std["cases"]["case_detection_rate"]),
         fmt(shift["cases"]["case_detection_rate"]), actor_label,
         f"{std['cases']['actors']} / {shift['cases']['actors']} actors"),
        ("alert_precision", fmt(std["cases"]["alert_precision"]),
         fmt(shift["cases"]["alert_precision"]), actor_label,
         f"{std['cases']['alerts']} / {shift['cases']['alerts']} alerts"),
        ("trace_coverage@2", fmt(std["cases"]["trace_coverage@2"]),
         fmt(shift["cases"]["trace_coverage@2"]), actor_label,
         f"{std['cases']['traceable_actors@2']} / "
         f"{shift['cases']['traceable_actors@2']} traceable actors"),
        ("trace_coverage@4", fmt(std["cases"]["trace_coverage@4"]),
         fmt(shift["cases"]["trace_coverage@4"]), actor_label,
         f"{std['cases']['traceable_actors@4']} / "
         f"{shift['cases']['traceable_actors@4']} traceable actors"),
        ("wallet_recall_broad (secondary)", fmt(std["cases"]["wallet_recall_broad"]),
         fmt(shift["cases"]["wallet_recall_broad"]),
         "broad wallet label (hops included)", "every illicit-pattern wallet"),
        ("stacker AUC", fmt(std["actor"]["auc"]), fmt(shift["actor"]["auc"]),
         actor_label,
         f"{std['actor']['positives']} / {shift['actor']['positives']} positive entities"),
        ("stacker AUC, old broad label", fmt(std["broad"]["auc"]),
         fmt(shift["broad"]["auc"]), "broad wallet label (hops included)",
         f"{std['broad']['positives']} / {shift['broad']['positives']} positive entities"),
        ("origin top-1 (split filter)", fmt(origin_rows["top1"]), "—",
         "observed_origin_ip per transaction", f"{origin_rows['n']} estimates"),
        ("origin cost-weighted score", fmt(origin_rows["cost_weighted_score"]), "—",
         "outcome costs +1 / +0.3 / -3 / 0", "split filter"),
        ("low_confidence_origin cutoff", fmt(origin_rows["cutoff"], 2), "—",
         "chosen on seed A, reported on seed B", f"seed A = {e['seed']}"),
        ("false positive rate, zero-attack", fmt(extra["zero"]["false_positive_rate"], 4),
         "—", "alerts per entity on traffic with nothing planted",
         f"{extra['zero']['entities']} entities, {extra['zero']['alerts']} alerts"),
        ("cluster ARI", fmt(float(extra["clusters"].iloc[0]["adjusted_rand_index"]), 4),
         fmt(float(extra["clusters"].iloc[1]["adjusted_rand_index"]), 4),
         "our wallet partition vs the generator's",
         f"{int(extra['clusters'].iloc[0]['wallets_scored'])} / "
         f"{int(extra['clusters'].iloc[1]['wallets_scored'])} wallets"),
        ("red-team detection rate, crimes only", "—",
         fmt(extra["redteam"]["detection_rate"]),
         "criminal injection raised at least one alert (section 7)",
         f"{extra['redteam']['crime_runs']} injections, shifted set"),
        ("red-team detection rate, all typologies", "—",
         fmt(extra["redteam"]["all_runs_rate"]),
         "includes the two patterns that are not crimes — see section 7",
         f"{extra['redteam']['completed']} injections"),
        ("red-team median transactions to detect", "—",
         f"{extra['redteam']['median_transactions_to_detect']}",
         ("the pattern's own transactions, in time order, before the first alert on it; "
         "crimes only; deterministic (wall-clock: section 7 footnote)"),
         f"{extra['redteam']['crime_detected']} detected"),
        ("attribution leads naming the true IP",
         fmt(extra["leads"]["standard"]), fmt(extra["leads"]["shifted"]),
         "leads shown beside an alert (not an AUC — see section 6)",
         f"{extra['leads']['n_standard']} / {extra['leads']['n_shifted']} leads"),
    ]
    return pd.DataFrame(rows, columns=["metric", f"standard (seed {e['seed']})",
                                       f"shifted (seed {e['seed']})",
                                       "label definition", "denominator"])


def build_report(cfg: dict, rebuild: bool = False) -> str:
    e = cfg["eval"]
    rates = e["observation_rates"]
    seed_a, seed_b = e["seed"], e["seed_b"]
    default_rate = e["default_rate"]
    standard = {rate: build(rate, False, cfg=cfg, rebuild=rebuild) for rate in rates}
    shifted = {rate: build(rate, True, cfg=cfg, rebuild=rebuild) for rate in rates}
    seed_b_sets = {"standard": build(default_rate, False, seed_b, cfg, rebuild=rebuild),
                   "shifted": build(default_rate, True, seed_b, cfg, rebuild=rebuild)}

    fusion = {"standard": fusion_eval.evaluate(standard[default_rate], cfg),
              "shifted": fusion_eval.evaluate(shifted[default_rate], cfg)}
    fusion_b = {name: fusion_eval.evaluate(ds, cfg) for name, ds in seed_b_sets.items()}

    provenance = gnn_provenance(cfg)
    clusters = clustering_eval.comparison(
        {"standard": standard[default_rate], "shifted": shifted[default_rate]}, cfg)
    quiet = zero_attack.evaluate(cfg, fusion["standard"]["stacker"], rebuild=rebuild)
    correlation = {
        name: correlation_eval.evaluate(
            {"standard": standard, "shifted": shifted}[name][default_rate],
            fusion[name]["bundle"], fusion[name]["alerted"], cfg)
        for name in ("standard", "shifted")}
    redteam = redteam_batch.run_batch(shifted[default_rate], cfg)
    saturated = {name: saturation.evaluate(fusion[name]["bundle"],
                                           fusion[name]["stacker"], cfg)
                 for name in ("standard", "shifted")}
    diagnosis = {name: saturation_diagnosis.evaluate(fusion[name]["bundle"],
                                                     fusion[name]["stacker"], cfg)
                 for name in ("standard", "shifted")}

    origin_results = origin.evaluate(standard, cfg)
    table = origin_results["table"]
    chosen, why = choose_default_estimator(table, cfg)
    comparison = origin.filter_comparison(standard[default_rate], cfg, chosen)
    split_row = comparison[comparison["filter"] == "split"].iloc[0]
    cutoff, sweep = origin.choose_cutoff(standard[default_rate], cfg, chosen)
    frame_b = origin.score_estimator(seed_b_sets["standard"], chosen, cfg, "split")["frame"]
    flag_b = origin.flag_quality(frame_b, cfg, cutoff)
    cost_b = origin.cost_score(frame_b, cfg, cutoff)

    parts: list[str] = []
    add = parts.append
    add("# Evaluation results\n")
    add(f"Generated by `python -m eval.report` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.\n")

    # --- summary ---------------------------------------------------------
    add("## Summary\n")
    add("Every number below states the dataset it came from, the seed, and what the\n"
        "label means. The unit of detection is the **actor** — one ground-truth\n"
        "illicit operation — pre-registered in `docs/detection_unit_protocol.md`\n"
        "before any of this was measured. Wallet recall is reported as secondary.\n")
    extra = {
        "zero": quiet, "clusters": clusters, "redteam": redteam,
        "leads": {
            "standard": correlation["standard"]["summary"]["leads_naming_the_true_ip"],
            "shifted": correlation["shifted"]["summary"]["leads_naming_the_true_ip"],
            "n_standard": correlation["standard"]["summary"]["leads_shown"],
            "n_shifted": correlation["shifted"]["summary"]["leads_shown"],
        },
    }
    add(md_table(summary_table(fusion, {"top1": float(split_row["top1"]),
                               "n": int(split_row["n"]),
                               "cost_weighted_score": float(split_row["cost_weighted_score"]),
                               "cutoff": cutoff}, cfg, extra)))
    add(f"\nOrigin figures: `{chosen}`, split filter, standard set, seed {seed_a}, "
        f"observation rate {default_rate}.\n")
    add(f"`low_confidence_origin` on **seed {seed_b}** (never used for tuning): "
        f"precision {flag_b['precision']}, recall {flag_b['recall']}, "
        f"accuracy with the flag clear {flag_b['accuracy_when_flag_clear']} against "
        f"{flag_b['accuracy_when_flagged']} when raised, cost-weighted score "
        f"{cost_b['cost_weighted_score']}.\n")

    add("\n### Read this before quoting anything above\n")
    add(WORSE.format(
        actors=fusion["standard"]["cases"]["actors"],
        actors_shifted=fusion["shifted"]["cases"]["actors"],
        actor_auc=fusion["standard"]["actor"]["auc"],
        actor_auc_shifted=fusion["shifted"]["actor"]["auc"],
        broad_auc=fusion["standard"]["broad"]["auc"],
        recall=fusion["standard"]["cases"]["wallet_recall_broad"],
        combined_top1=float(comparison[comparison["filter"] == "combined"]["top1"].iloc[0]),
        split_top1=float(split_row["top1"]),
        off_top1=float(comparison[comparison["filter"] == "off"]["top1"].iloc[0]),
        wrong_combined=int(comparison[comparison["filter"] == "combined"]
                           ["wrong_uninvolved_third_party"].iloc[0]),
        wrong_split=int(split_row["wrong_uninvolved_third_party"]),
        abstained_split=int(split_row["abstained"]),
        n=int(split_row["n"]),
        flag_precision=flag_b["precision"]))

    add(f"\n**GNN provenance: {provenance['status']}.** {provenance['detail']}.\n")
    if not provenance["usable"]:
        add("Every `gnn_score` below is therefore either zero or not to be trusted, and "
            "is\nmarked as such rather than quietly folded into the stack. The GNN is "
            "trained on\n`eval.gnn_train_seed`; the check that enforces it lives in "
            "`eval.report.gnn_provenance`.\n")

    add("\n**Canonical setup.** Seed A = "
        f"{seed_a}, seed B = {seed_b}, {e['n_actors']} actors, "
        f"{e['n_transactions']} transactions, observation rates {rates}, decisions "
        f"made at rate {default_rate}. These live in `config.yaml` under `eval:`. "
        "Seed B is\nonly ever reported, never used to choose a threshold.\n")
    add("The **shifted** set applies `--shifted`: jittered peel ratios, deeper chains,\n"
        "longer windows and interleaved CoinJoins. Where both are shown, **the shifted\n"
        "number is the headline** — it is the one that asks whether a detector learned the\n"
        "pattern or memorised our parameters.\n")

    # --- detection -------------------------------------------------------
    add("\n## 1. Detection, scored on actors\n")
    fusion_section(add, fusion["standard"], "Standard set", standard[default_rate])
    fusion_section(add, fusion["shifted"], "Shifted set", shifted[default_rate])

    add(f"\n### Replication on seed {seed_b}\n")
    add("The canonical dataset contains only a handful of illicit operations, so a "
        "case\nrate moves in steps of 20 percentage points. Seed B is reported to show "
        "whether\nthe standard-set numbers are a property of the system or of one "
        "draw.\n\n")
    replication = pd.DataFrame([
        {"set": name, "seed": seed_b,
         **{k: v for k, v in result["cases"].items()
            if k in ("actors", "case_detection_rate", "alert_precision",
                     "trace_coverage@2", "trace_coverage@4", "wallet_recall_broad")}}
        for name, result in fusion_b.items()])
    add(md_table(replication))

    # --- origin ----------------------------------------------------------
    add("\n## 2. Origin estimation\n")
    display = table[["rate", "estimator", "n", "top1", "top3", "conditional_top1",
                     "ceiling", "ceiling_unconditional", "brier"]].rename(columns={
                         "top1": "top-1", "top3": "top-3",
                         "conditional_top1": "top-1 given observed",
                         "ceiling": "ceiling (multi-hop)",
                         "ceiling_unconditional": "ceiling (all txs)",
                         "brier": "Brier"})
    add(md_table(display))
    add("`top-1 given observed` is accuracy restricted to transactions whose true origin "
        "appears\nin the observed tree at all. It separates \"the estimator is wrong\" "
        "from \"the answer\nwas never in the data\".\n")
    add("\nThe ceiling has **two denominators and they are not interchangeable** — "
        "quoting one\nas the other is how two different ceiling figures ended up in "
        "circulation:\n\n"
        "* **ceiling (multi-hop)** — of the transactions an estimator is actually asked\n"
        "  about, those observed at more than one relay, how often the true origin is\n"
        "  among the observed addresses. This is what bounds the accuracy columns beside\n"
        "  it.\n"
        "* **ceiling (all txs)** — of *every* transaction in the dataset. Lower, because\n"
        "  a transaction seen at a single relay is usually seen somewhere that is not its\n"
        "  source. This is the one that describes the evidence an operator has.\n")

    configured = cfg["engines"]["propagation"]["estimator"]
    add(f"\n**Protocol decision** (`docs/origin_eval_protocol.md`): the default "
        f"estimator is **`{chosen}`** — {why}.\n")
    add(f"Configured default is `{configured}`"
        + (".\n" if configured == chosen else f" — **needs updating to `{chosen}`**.\n"))

    add("\n### The origin filter, split three ways\n")
    add(f"`{chosen}`, standard set, rate {default_rate}. `off` applies no class "
        "weighting;\n`combined` is the old filter (relay, Tor and hosting all "
        "penalised in the ranking);\n`split` penalises relays only and reports Tor and "
        "hosting as anonymized entry\npoints with a reduced attribution confidence "
        "(`attribution_confidence_factor`).\n\n")
    add(md_table(comparison))
    add("\nThe `off` row abstains on every estimate, which is not a bug: confidence is "
        "the\nwinner's share of the total score, so with nothing down-weighted every "
        "share falls\nbelow the cutoff. Accuracy cannot see that — it is what the "
        "cost-weighted score is\nfor.\n")
    add("\nOutcome costs are pre-registered in "
        "`engines.propagation.origin_filter.cost_weights`:\n`correct_actionable` +1, "
        "`correct_infrastructure` +0.3, `wrong_uninvolved_third_party` -3,\n"
        "`abstained` 0.\n")

    add("\n### Every estimator under every filter\n")
    add(md_table(origin.class_weight_ablation(standard[default_rate], cfg)))

    add("\n### Every estimator, every rate, filter on and off\n")
    add("The full cross. The table above fixes the rate and varies the filter; the one\n"
        "at the top of this section fixes the filter and varies the rate. Neither says\n"
        "whether the filter still earns its place as observation gets sparser, which is\n"
        "the regime a real deployment is in. `share of ceiling` is top-1 divided by the\n"
        "fraction of transactions whose true origin was in the tree at all — the only\n"
        "comparison that is fair across rates.\n\n")
    add(md_table(origin.rate_filter_cross(standard, cfg)))
    add("\nThese figures used to exist only as printed output from "
        "`tests/test_propagation.py`.\nThey are here now, and that test asserts against "
        "`eval.origin` rather than\nre-deriving them, so the two cannot drift apart "
        "again.\n")

    add("\n### Calibration\n")
    frame = origin_results["frames"].get(chosen)
    if frame is not None:
        add(f"Reliability of `{chosen}` at rate {default_rate}: stated confidence "
            "against how often it was right.\n\n")
        add(md_table(origin.reliability(frame, e["calibration_bins"])))

    add("\n### `low_confidence_origin`\n")
    add(f"Renamed from `origin_likely_unobserved` (API, alerts, config, docs). Cutoff "
        f"chosen\non **seed {seed_a}** by the pre-registered rule — the value "
        "maximising the\ncost-weighted score, ties to the lower cutoff:\n\n")
    add(md_table(sweep))
    add(f"\n**Chosen cutoff: {cutoff}.** Reported unchanged on **seed {seed_b}**:\n\n")
    add(md_table(pd.DataFrame([flag_b])))
    add("\n" + md_table(pd.DataFrame([cost_b])))
    add("\n`precision` and `recall` are against \"the true origin was not in the "
        "observed tree\nat all\" — the claim the old name made. They are low because "
        "that event is rare.\nThe accuracy split (flag clear vs. raised) is what the "
        "flag is for and what it\ndelivers.\n")

    # --- 3. zero attack --------------------------------------------------
    add("\n## 3. False positives, on traffic with nothing in it\n")
    add("The same pipeline, the same threshold, the same fitted stacker — run on the\n"
        "canonical setup with `generator.pattern_mix` set to `normal` only. Nothing is\n"
        "planted. Every alert here is a false positive, and this is the number an\n"
        "operator lives with: recall says what we catch, this says what we cost.\n")
    add(f"\n{quiet['dataset']}. {quiet['transactions']} transactions, "
        f"{quiet['entities']} entities, planted patterns: "
        f"{quiet['planted_patterns'] or 'none'}.\n")
    add(f"\n**{quiet['alerts']} alerts at threshold {quiet['threshold']} — a false "
        f"positive rate of {quiet['false_positive_rate']:.4f} per entity.** "
        f"The rules engine fired {quiet['rule_alerts']} times; "
        f"{quiet['watchlist_seeds']} entities were watchlist seeds.\n\n")
    add(md_table(quiet["scores"]))
    if quiet["alerts"]:
        add("\n**Which signal put them there:**\n\n")
        add(md_table(quiet["by_signal"]))
        add("\n**The worst of them, with the reason the system gave:**\n\n")
        add(md_table(quiet["top"]))
    add("\n" + ZERO_ATTACK_NOTE)

    # --- 3b. saturation --------------------------------------------------
    add("\n## 4. Does the ranked queue rank?\n")
    add(SATURATION_PREAMBLE)
    for name in ("standard", "shifted"):
        block = saturated[name]
        add(f"\n### {name.capitalize()} set\n")
        if not block["alerts"]:
            add("No alerts fired, so there is nothing to rank.\n")
            continue
        add(f"{block['alerts']} alerts above the {block['threshold']} threshold. "
            f"Scores run from **{block['min']}** to **{block['max']}** — a spread of "
            f"**{block['spread']}**.\n")
        add(f"\n**{block['at_exactly_one']} alerts score exactly 1.000.** "
            f"Across the whole queue there are **{block['distinct_values']} distinct "
            f"values** at the three decimals the console prints; in the top "
            f"{block['top_n']} there are **{block['distinct_in_top']}**. The most "
            f"common single value is {block['most_common_value']:.3f}, shared by "
            f"{block['share_at_most_common']:.1%} of alerts.\n")
        add("\n**Deciles:**\n\n")
        add(md_table(block["deciles"], floats=4))
        add(f"\n**The head of the queue** — the composite values an analyst sorting "
            f"by risk sees:\n\n")
        add(md_table(block["top_values"]))
        add(f"\n**What the tiebreakers recover.** The queue no longer sorts on the "
            f"composite alone (`fusion/ordering.py`). Against "
            f"{block['distinct_values']} distinct composite value(s), the full sort key "
            f"gives **{block['queue_distinct_keys']} distinct keys** over "
            f"{block['alerts']} alerts — a total order by construction, since the entity "
            f"id ends it. The number that matters is the one before that backstop: "
            f"**{block['queue_distinct_evidential_keys']} distinct keys from evidence "
            f"alone**, and **{block['queue_distinct_evidential_in_top']} in the top "
            f"{block['top_n']}** where the composite gave "
            f"{block['distinct_in_top']}. "
            f"{block['queue_resolved_by_id_alone']} of the {block['alerts']} alerts are "
            f"separated only by the entity id: deterministic, but not meaningful.\n")

    add("\n### Where the resolution is lost\n")
    add(DIAGNOSIS_PREAMBLE)
    for name in ("standard", "shifted"):
        d = diagnosis[name]
        if not d.get("alerts"):
            continue
        add(f"\n#### {name.capitalize()} set\n")
        add(f"{d['alerts']} alerts print **{d['distinct_displayed_scores']} distinct "
            f"scores** — but they have **{d['distinct_input_vectors']} distinct input "
            f"vectors**. Only {d['alerts_sharing_an_input_vector']} alerts share their "
            f"inputs with another (largest identical group: "
            f"{d['largest_identical_input_group']}). "
            f"**{d['distinguishable_but_collapsed']} distinct input vectors are "
            f"collapsed onto a shared displayed score.**\n")
        add(f"\nThe pre-sigmoid logit runs from **{d['logit_min']}** to "
            f"**{d['logit_max']}** — a spread of {d['logit_spread']} in log-odds — "
            f"yet takes only **{d['logit_distinct']} distinct values**. "
            f"{d['above_flat_logit']} of {d['alerts']} alerts sit above "
            f"{d['flat_logit']}, where the logistic curve is flat to three decimals.\n")
        add("\n**Each input signal among the alerts:**\n\n")
        add(md_table(d["inputs"], floats=4))
        add("\n**The logit, by decile, and what the sigmoid does with it:**\n\n")
        add(md_table(d["logit_deciles"], floats=6))
        if len(d["identical_groups"]):
            add("\n**Alerts with genuinely identical inputs** — these no "
                "transformation can separate:\n\n")
            add(md_table(d["identical_groups"], floats=4))
    d = diagnosis["standard"]
    weighted = [r for r in d["inputs"].to_dict("records")
                if r["signal"] in ("rule_score", "taint_score")]
    add("\n" + DIAGNOSIS_VERDICT.format(
        alerts=d["alerts"], vectors=d["distinct_input_vectors"],
        logits=d["logit_distinct"], spread=d["logit_spread"],
        flat=d["flat_logit"], displayed=d["distinct_displayed_scores"],
        anomaly_distinct=next(r["distinct"] for r in d["inputs"].to_dict("records")
                              if r["signal"] == "anomaly_score"),
        coarse=" and ".join(f"`{r['signal']}` takes {r['distinct']} distinct values"
                            for r in weighted),
        threshold=cfg["fusion"]["alert_threshold"]))

    # --- 5. clustering ---------------------------------------------------
    add("\n## 5. Cluster quality\n")
    add("Every detection number in this report is scored per entity, and an entity is\n"
        "whatever `graph/clustering.py` decided. The Adjusted Rand Index compares our\n"
        "partition of the wallets against the generator's true one — adjusted for\n"
        "chance, so 0 is what random grouping scores and 1 is exact agreement. Cluster\n"
        "*names* are arbitrary on both sides, so only the partition can be compared.\n\n")
    add(md_table(clusters, floats=4))
    add("\n" + CLUSTER_NOTE)

    # --- 6. correlation --------------------------------------------------
    add("\n## 6. Attribution leads, measured as attribution\n")
    add(CORRELATION_PREAMBLE)
    for name in ("standard", "shifted"):
        block = correlation[name]
        summary = block["summary"]
        add(f"\n### {name.capitalize()} set\n")
        add(f"{summary['links']} links over {summary['entities_with_a_link']} entities; "
            f"{summary['leads_shown']} leads shown across "
            f"{summary['alerts_with_a_lead']} alerts.\n")
        if summary["leads_shown"]:
            add(f"\n**Of the leads an analyst is shown, "
                f"{summary['leads_naming_the_true_ip']:.3f} name the actor's true "
                f"broadcast address** and "
                f"{summary['leads_naming_the_observed_ip']:.3f} name the address the "
                f"broadcast entered the network from. The true address was observable "
                f"at all for {summary['true_ip_observable']:.3f} of them — that is the "
                f"ceiling, not a failure of the engine.\n")
        add("\n**Every scored link, by score band:**\n\n")
        add(md_table(block["by_score"]))
        add("\n**Leads actually shown beside an alert, by score band:**\n\n")
        add(md_table(block["shown_by_score"]))
        add("\n**By rank within an alert** — rank 1 should beat rank 3, or the "
            "ordering is decoration:\n\n")
        add(md_table(block["by_rank"]))
        add("\n**By how many distinct transactions the link rests on:**\n\n")
        add(md_table(block["by_observations"]))

    # --- 7. red team -----------------------------------------------------
    add("\n## 7. Red team, 50 injections\n")
    add(REDTEAM_PREAMBLE)
    add(f"\n**{redteam['crime_detected']} of {redteam['crime_runs']} criminal "
        f"injections were detected — {redteam['detection_rate']:.3f}** at threshold "
        f"{redteam['threshold']}. **Median transactions to detect: "
        f"{redteam['median_transactions_to_detect']}** of the pattern's own transactions "
        f"(median share of the injection: {redteam['median_fraction_to_detect']}), fed "
        "in timestamp order to the same incremental update the endpoint runs. A count, "
        "so it does not depend on the machine; wall-clock is a footnote below."
        + (f" {redteam['failed']} run(s) failed outright.\n" if redteam["failed"]
           else "\n"))
    add(f"\nOver **all {redteam['completed']}** injections including the two "
        f"non-crime patterns the figure is {redteam['all_runs_detected']} detected, "
        f"{redteam['all_runs_rate']:.3f} — shown so the exclusion below cannot be "
        "mistaken for\nsomething being hidden.\n")
    add("\n" + NON_ACTOR_NOTE)
    add("\n**Per typology:**\n\n")
    add(md_table(redteam["per_typology"]))
    add("\n**By broadcast route** — what the network side could recover:\n\n")
    add(md_table(redteam["by_broadcast"]))
    add(REDTEAM_WALLCLOCK)

    add("\n### The two patterns that are not crimes\n")
    add("Scored as what they are. There is nothing to catch, so \"detection rate\" is "
        "not\nthe question: an alert here is a **false positive**, and the pass is "
        "silence plus\na clustering that did the right thing — CoinJoin participants "
        "kept in separate\nentities, one actor's wallets pulled together.\n\n")
    add(md_table(redteam["non_actor"]))
    if len(redteam["misses"]):
        add(f"\n### The misses\n")
        add("Criminal injections that raised nothing, with every engine's score for the\n"
            "injected entities against the threshold they did not clear. Up to three\n"
            "entities per missed injection, closest first. The two non-crime patterns are\n"
            "not in this table — they are not misses.\n\n")
        add(md_table(redteam["misses"].head(40)))
        add("\nOne thing this table says loudly: **the fused score does not move with the "
            "anomaly\nscore.** Injections with `anomaly` above 0.9 land on the same fused "
            "value as ones\nat 0.14, because the stacker's coefficient for "
            "`anomaly_score` is 0 (see the\nablation in section 1) — the signal is "
            "carried but not used. That is the fitted\nmodel's verdict on it, not a "
            "bug, and it is why the anomaly engine is the first\nplace to look if "
            "these detection rates need to improve.\n")
        if len(redteam["misses"]) > 40:
            add(f"\n_({len(redteam['misses'])} rows in total; the first 40 are shown.)_\n")
    else:
        add("\nNo injection went undetected.\n")
    if len(redteam["errors"]):
        add("\n**Runs that failed:**\n\n")
        add(md_table(redteam["errors"]))

    # --- 8. the anomaly engine -------------------------------------------
    add("\n## 8. The anomaly engine contributes nothing\n")
    add(anomaly_section(fusion, redteam))

    # --- 9. ground truth on real relay data ------------------------------
    add("\n## 9. Origin accuracy against known truth\n")
    # Imported here: origination imports eval, not the other way round.
    from origination import evaluate as origination_eval
    from origination import report as origination_report

    origination = origination_eval.run(cfg, rebuild)
    ground_truth_section(add, ground_truth.evaluate(cfg, rebuild), cfg,
                         origination_report.section(origination, md_table))
    origination_report.write_doc(origination, md_table, cfg["origination"]["results_doc"])

    # --- 10. the validity layer ------------------------------------------
    add("\n## 10. The validity layer: when attribution is invalid\n")
    from analysis import evaluate as validity_eval

    validity = validity_eval.run(cfg, rebuild, base=origination)
    # The pre-registration and P6's frozen numbers first, verbatim, one level down.
    add("\n" + validity_eval.preserved(cfg["validity"]["results_doc"])
        .replace("\n### ", "\n#### ").replace("\n## ", "\n### ")
        .replace("## Metric revision", "### Metric revision", 1))
    add("\n### Results — metric revision\n\n")
    add(validity_eval.section(validity, md_table).replace("\n### ", "\n#### "))
    validity_eval.write_doc(validity, md_table, cfg["validity"]["results_doc"])

    # --- 11. the reverse direction ---------------------------------------
    add("\n## 11. Peer profiles: coverage\n")
    add(profile_section(cfg, origination))

    # --- 12. wallet fingerprints ------------------------------------------
    add("\n## 12. Wallet-construction fingerprints\n")
    add(fingerprint_section(cfg, rebuild))

    # --- 13. actors ------------------------------------------------------------
    add("\n## 13. Actors: the queue as triage\n")
    add(actor_section({"standard": standard[default_rate], "shifted": shifted[default_rate]},
                      {n: fusion[n]["stacker"] for n in ("standard", "shifted")}, cfg))

    add(CLOSING)
    return "\n".join(parts)


#: Wall-clock time-to-detect, measured once, by hand, not regenerated: seconds
#: depend on the machine's power state and load (docs/VALIDITY.md, red-team
#: timing), so the headline above is a transaction count instead.
REDTEAM_WALLCLOCK = """
*Footnote — wall-clock.* Measured once on 2026-09-26 on the development laptop,
`eval.redteam_batch.run_batch` on the shifted dataset, a discarded warm-up batch
first: median inject-to-alert **2.41 s** in the warm-up (on AC power throughout)
and **2.52 s** in the measured batch, over the 20 detected criminal injections. The
AC adapter read offline when the measured batch finished, so it is not a clean
on-AC figure; treat both as indicative. The transaction count above was identical
in both batches (median 11.5), which is why it is the headline. Copied through, not
regenerated.
"""

def actor_section(datasets: dict, stackers: dict, cfg: dict) -> str:
    """Section 13: the served demo's distribution shift, then the actor queue
    against the entity queue on each dataset (eval/actor_queue.py)."""
    from eval import actor_queue
    out = [demo_shift_section(cfg)]
    results = actor_queue.evaluate(datasets, stackers, cfg)
    out.append(
        "\n### Actor queue vs entity queue\n\n"
        "`condition=\"simulated\"`, cross-topology: the generator's gossip network "
        "(relay_hub, 500 nodes, relay share 0.08) is not among the origination corpus's "
        "training configurations. Both queues come from one fusion bundle and one fitted "
        "stacker per dataset; the only difference is the unit (docs/ACTORS.md). An item "
        "*finds* an illicit operation (`eval.actors.actors_of`) when it holds any of its "
        "wallets. The join rule was fixed before this evaluation was run.\n")
    for name, r in results.items():
        q = r["quality"]
        out += [f"\n#### {name}\n\n", md_table(r["table"]), "\n",
                md_table(pd.DataFrame([{"measure": k, "value": str(v)} for k, v in q.items()])),
                "\n"]
    std, shf = results["standard"]["table"], results["shifted"]["table"]

    def better(t):
        before, after = t.iloc[0], t.iloc[1]
        cols = [c for c in t.columns if c.startswith(("precision@", "recall@"))]
        gain = any((after[c] or 0) > (before[c] or 0) for c in cols)
        work = [c for c in t.columns if c.startswith("reviewed to find")]
        counted = lambda v: not isinstance(v, str) and pd.notna(v)
        less = any(counted(after[c]) and counted(before[c]) and after[c] < before[c]
                   for c in work)
        return gain, less
    verdicts = {n: better(t) for n, t in (("standard", std), ("shifted", shf))}
    lines = []
    for n, (gain, less) in verdicts.items():
        lines.append(f"{n}: precision/recall@k {'improves somewhere' if gain else 'does not improve'}, "
                     f"workload {'drops somewhere' if less else 'does not drop'}")
    out.append("\n**Verdict, plainly.** " + "; ".join(lines) + ". These datasets hold only "
               f"{results['standard']['quality']['illicit operations present']} and "
               f"{results['shifted']['quality']['illicit operations present']} illicit "
               "operations, and nearly every alert already holds an illicit wallet, so "
               "precision@k saturates for both queues and small differences are within one "
               "operation. What the actor queue changes is mostly the count: "
               f"{results['standard']['quality']['alert count reduction']:.1%} and "
               f"{results['shifted']['quality']['alert count reduction']:.1%} fewer items, at "
               "the wrong-merge rates above. It is not shown to improve triage beyond that "
               "on this data.\n")
    return "".join(out)


def demo_shift_section(cfg: dict) -> str:
    """The served demo dataset against the corpus it is compared with."""
    import json as _json
    import math
    from collections import Counter

    from features import fingerprint as F
    from graph.builder import iter_transactions, load
    from origination import evaluate as OE
    from origination.model import KEY, OriginationModel

    out = ["\n### The served demo dataset is shifted from the corpora\n\n"]
    gt = _json.loads((Path(cfg["ingest"]["input_dir"]) / "ground_truth.json").read_text())["transactions"]
    matrix = pd.read_parquet(cfg["features"]["relay_path"])
    model = OriginationModel.load(cfg["origination"]["model_path"])
    keys = matrix[KEY].drop_duplicates().itertuples(index=False)
    truth = pd.Series({(c, t): gt[t]["observed_origin_ip"] for c, t in keys if t in gt})
    decided = OE.label_coinjoins(model.decide(matrix, truth, cfg),
                                 {(c, t) for c, t in truth.index if gt[t]["pattern"] == "coinjoin"})
    row = OE._row(OE.MODEL, decided, cfg, model.cutoff, "served demo capture")
    cols = ["n", "top1", "abstention rate", "acc if answered", "ceiling (origin observed)",
            "cost_weighted_score"]
    out += [("The origination model on the served demo capture (a pooled collector over "
            "the relay-hop log), scored as the corpus rows in section 9 are. Its "
            "cross-topology corpus row: top1 0.274, abstention 0.808, accuracy if "
            "answered 0.906, cost 0.120, ceiling 0.279. The demo's ceiling is far higher "
            "because a pooled collector sees the origin far more often, so the absolute "
            "numbers are not comparable; relative to its ceiling, top1 is "
            f"{row['top1'] / row['ceiling (origin observed)']:.2f} here and "
            f"{0.274 / 0.279:.2f} on the corpus. Origination does not degrade on the demo, "
            "so the generators were not aligned for it.\n\n"),
            md_table(pd.DataFrame([{k: row.get(k) for k in cols}])), "\n"]

    fp = F.load_model(cfg)
    worst = Counter()
    for tx in iter_transactions(load(None, cfg)):
        t = F.tells(F.View.of_tx(tx))
        a = fp.classify(t, cfg)
        if not a.get("novelty", {}).get("novel"):
            continue
        top = a["ranked"][0]["label"]

        def surprise(tell, t=t, top=top):
            v = t[tell]
            k = len(fp.vocab[tell]) + (v not in fp.vocab[tell])
            c = fp.counts.get(top, {}).get(tell, {}).get(v, 0)
            return -math.log((c + fp.meta["alpha"]) / (fp.totals[top].get(tell, 0) + fp.meta["alpha"] * k))
        tell = max((x for x in t if t[x] is not None and x in fp.vocab), key=surprise)
        kind = "ordinary payment" if gt[tx.txid]["pattern"] == "normal" else "typology"
        worst[(kind, tell)] += 1
    out += [("\nWhat the fingerprint novelty check trips on in the demo, by the tell that "
            "was least supported under the named pattern:\n\n"),
            md_table(pd.DataFrame([{"transaction": k, "worst tell": t, "flagged": n}
                                   for (k, t), n in worst.most_common()])),
            ("\n`script_mix`: `generator.main` gives each actor a script type from "
            "`generator.script_types`, independent of its wallet profile, while the corpus "
            "draws input types from the profile's `input_types`. `io_shape`: the typologies "
            "build 1-in-1-out and wide fan-out transactions the corpus's three shapes "
            "(payment, batch, CoinJoin) never contain. Both are generator differences, not "
            "model faults, and only the fingerprint depends on them.\n")]
    return "".join(out)


FINGERPRINT_OMISSIONS = (
    "The profiles are this simulator's stand-ins for the families they are named after, "
    "built from the tells in docs/FINGERPRINTS.md, several of which are assumptions about "
    "the real software rather than documented behaviour. The classifier is fitted to the "
    "same profiles it is scored on, so these numbers measure how well it recovers the "
    "simulator's own construction rules, not how well it would identify real wallets. "
    "Transaction shapes are the corpus's three (payment, batch, CoinJoin); vsize is the "
    "standard single-key estimate, exact here because the generator uses the same table.")


#: Above this pooled harmful-mislabel rate (all tells, open-set), fingerprints
#: stay out of the console by default. Pre-registered in docs/FINGERPRINTS.md.
FINGERPRINT_HARM_LIMIT = 0.10

#: §12's transfer figures from the generator before P8.1 (commit 74c404e). That
#: generator no longer exists, so they are copied through, not regenerated.
TRANSFER_PRE_8_1 = """
#### Typologies rebuilt in full under profiles (pre-P8.1 generator, 74c404e)

Copied from commit 74c404e's report, not regenerated: that generator is gone. It
rebuilt every typology transaction with the full profile, including fee rounding,
dust-dropping and paying change back to an input. That gave typology transactions
every tell but broke peel chains (their amounts and change addresses no longer
linked hop to hop), which is why P8.1 replaced it. Same seed, same model.

| condition | typology | transactions | unknown rate | accuracy if answered |
| --- | --- | --- | --- | --- |
| simulated | coinjoin | 88 | 0.511 | 0.767 |
| simulated | layering | 123 | 0.081 | 0.434 |
| simulated | normal | 2523 | 0.080 | 0.898 |
| simulated | ransomware_collector | 176 | 0.528 | 0.795 |
| simulated | same_actor_cluster | 90 | 0.044 | 0.442 |

| true profile | core_like | electrum_like | legacy_naive | coordinator_coinjoin | batch_withdrawal | unknown |
| --- | --- | --- | --- | --- | --- | --- |
| coordinator_coinjoin | 0 | 0 | 0 | 33 | 10 | 45 |
| core_like | 976 | 113 | 0 | 0 | 1 | 190 |
| electrum_like | 188 | 513 | 3 | 1 | 0 | 96 |
| legacy_naive | 1 | 3 | 748 | 55 | 0 | 24 |
"""

def fingerprint_section(cfg: dict, rebuild: bool = False) -> str:
    """Section 12: per-class precision/recall, confusion matrices and unknown
    rates on the fingerprint corpus's test sets, the CoinJoin agreement with
    the validity layer, and a transfer check on the generator's own typologies."""
    from features import fingerprint as F

    fitted = F.fit(cfg, rebuild)
    result = F.evaluate(fitted, cfg)
    transfer = F.transfer(fitted["model"], cfg)
    lopo = F.leave_one_profile_out(fitted["truth"], cfg)
    cost = F.in_distribution_cost(fitted, cfg)
    pooled = lopo[lopo["held-out profile"] == "all (pooled)"].set_index(["condition", "model"])
    harm = {(c, m): pooled.loc[(c, m), "harmful mislabel rate"]
            for c in (F.FULL, F.STRUCTURAL) for m in (F.OPEN_SET, F.CLOSED_SET)}
    high = harm[(F.FULL, F.OPEN_SET)] > FINGERPRINT_HARM_LIMIT
    shown = cfg["features"]["fingerprint"].get("console_display", True)
    note = f"\n*`condition=\"simulated\"`.* {FINGERPRINT_OMISSIONS}\n"
    f = cfg["features"]["fingerprint"]
    out = [
        (f"Corpus `{fitted['corpus']}` (`origination/manifest_fingerprint.json`): the validity "
        "variant with every transaction built under a wallet-construction profile "
        "(`generator/wallets.py`). Naive Bayes over the tells, fitted on the training "
        "captures, isotonic-calibrated on the calibration captures, scored on the within- "
        "and cross-topology test captures. **full**: every tell the corpus records. "
        "**structural**: version, nLockTime and nSequence masked — what a relay log in the "
        "NTRO schema shows — with its own calibration. The answer is `unknown` under "
        f"calibrated confidence {f['unknown_below']} or with fewer than {f['min_tells']} "
        "observable tells (config.yaml, with the rationale).\n"),
        ("\n### Generalization: leave one profile out\n\n"
        "**This is the generalization result.** Every other table here scores the model on "
        "profiles it was fitted to. Here each profile is held out in turn. The model is "
        "fitted and calibrated (isotonic maps and novelty thresholds) on the other four "
        "and asked about the held-out one's cross-topology test transactions. Scoring is "
        "pre-registered (docs/FINGERPRINTS.md, \"Open-set revision\"). `unknown` is right. "
        "A **shared pattern** names a known pattern whose defining tells (at least 80% of "
        "its training rows) the transaction shows. A **harmful mislabel** names one it "
        "contradicts. `P8.1 (no novelty check)` is the same fitted model without the "
        "check: P8.1's decision rule, scored the same way.\n\n"
        f"**Harmful-mislabel rate on never-seen profiles: "
        f"{harm[(F.FULL, F.OPEN_SET)]:.3f} with all tells, "
        f"{harm[(F.STRUCTURAL, F.OPEN_SET)]:.3f} structure only** (pooled over the "
        f"five hold-outs; lower is better). P8.1: {harm[(F.FULL, F.CLOSED_SET)]:.3f} and "
        f"{harm[(F.STRUCTURAL, F.CLOSED_SET)]:.3f}. "
        + ((f"That is above the pre-registered {FINGERPRINT_HARM_LIMIT}: the fingerprint "
            "still gives many never-seen constructions a known label they contradict. "
            "Under the pre-registered rule the console hides fingerprints unless "
            "`features.fingerprint.console_display` is turned on"
            + (" (it is off)." if not shown else
               " — **but config.yaml has it on, contrary to the rule.**") + " ")
           if high else
           (f"That is within the pre-registered {FINGERPRINT_HARM_LIMIT}, so the console "
            "shows fingerprints by default. "))
        + "Still simulated: the held-out construction is another of this simulator's "
        "profiles, so real unseen software may sit closer to or further from the known "
        "ones.\n\n"),
        md_table(lopo), note,
        ("\n### What the novelty check costs on known profiles\n\n"
        "The five-profile model on the known-profile test sets, with and without the "
        "novelty check. Its threshold is the 99th percentile of the novelty score on the "
        "calibration rows, per condition, fixed before any evaluation.\n\n"),
        md_table(cost), note,
        "\n### Unknown rate and accuracy when answered\n\n", md_table(result["unknown"]), note,
        ("\n### Per class\n\n`recall` counts an unknown as a miss; `recall if answered` does "
        "not.\n\n"), md_table(result["per_class"]), note,
    ]
    for label in ("cross-topology, full", "cross-topology, structural", "within-topology, full"):
        out += [(f"\n### Confusion matrix — {label}\n\nRows are the generator's profile, "
                "columns the fingerprint.\n\n"),
                md_table(result["confusion"][label].reset_index()), note]

    agree = result["agreement"]
    dis = result["disagreements"]
    cross = dis[dis["test set"] == "cross-topology, full"]

    def n(kind, truth=None, said=None):
        m = cross["disagreement"] == kind
        if truth:
            m &= cross["true profile"] == truth
        if said:
            m &= cross["fingerprint said"] == said
        return int(cross.loc[m, "transactions"].sum())

    out += [
        ("\n### CoinJoin: the fingerprint against the validity layer\n\n"
        "`coordinator_coinjoin` named by the fingerprint, beside `analysis.validity`'s "
        "COINJOIN detector (`graph.clustering.is_coinjoin`) on the same transactions. They "
        "are compared, not reconciled: each keeps its own answer.\n\n"), md_table(agree), note,
        "\n#### Where they disagree\n\n", md_table(dis), note,
        ("\n**What the disagreements are** (cross-topology, full). "
        f"{n('detector only', 'batch_withdrawal')} are batched withdrawals the detector calls "
        "a CoinJoin — its documented false positive (an equal-value batch paying no more equal "
        "amounts than it spends inputs, docs/VALIDITY.md) — which the fingerprint names "
        f"correctly in {n('detector only', 'batch_withdrawal', 'batch_withdrawal')} cases, "
        "from the single change output, the round fee rate and the payees' mixed script types. "
        f"{n('detector only', 'coordinator_coinjoin')} are true CoinJoins the detector finds and "
        f"the fingerprint does not ({n('detector only', 'coordinator_coinjoin', 'unknown')} "
        f"unknown, {n('detector only', 'coordinator_coinjoin', 'batch_withdrawal')} called a "
        "batch): rounds where few participants took change look like a batch to the tells. "
        f"{n('fingerprint only')} are transactions only the fingerprint calls a CoinJoin, "
        f"{n('fingerprint only', 'batch_withdrawal')} of them batches the detector's participant "
        "rule (no more equal outputs than inputs) correctly declines. Neither is forced to "
        "agree with the other: the validity layer's verdict governs what an origin answer may claim, and the "
        "fingerprint is shown beside it.\n"),
        ("\n### Transfer: the generator's own typologies\n\n"
        f"The model above, on a `generator.main --wallet-profiles` dataset (seed "
        f"{transfer['seed']}, {transfer['transactions']} transactions): the same profiles, "
        "on peel chains, layering fan-outs and same-actor spends the corpus never shows. "
        "Two generator versions, two different shifts. Neither is a generalization claim "
        "(that is the leave-one-profile-out table above): both are the simulator's own "
        "profiles, scored by a model fitted to them.\n\n"
        "#### Typologies with chain-neutral tells only (current generator, P8.1)\n\n"
        "Ordinary payments get every tell. Typology transactions get version, nLockTime, "
        "nSequence and ordering, plus an input script type only where no other "
        "transaction sees the inputs. Their fees and change are the typology's own. So "
        "they carry fewer profile tells than the corpus does, and the figure reflects "
        "how the dataset is generated now.\n\n"),
        md_table(transfer["by_typology"]), note,
        "\n", md_table(transfer["confusion"].reset_index()), note,
        TRANSFER_PRE_8_1,
    ]
    return "".join(out)


def profile_section(cfg: dict, origination: dict) -> str:
    """What share of peers get each part of a profile (engines/correlation/
    profile.py), over what the API serves and over the base corpus's
    cross-topology test captures."""
    from engines.correlation import profile

    served = profile.build_sources(cfg)
    matrix = origination["parts"]["cross_test"]
    corpus = profile.Sources(matrix=matrix, answers=profile.origination_answers(
        origination["model"], matrix, cfg, shapes=origination["shapes"]))
    columns = {"served (relay-hop dataset + relay matrix)": profile.coverage(served, cfg),
               f"base corpus `{origination['corpus']}`, cross-topology test captures":
               profile.coverage(corpus, cfg)}
    rows = []
    for measure in next(iter(columns.values())):
        row = {"condition": "simulated", "measure": measure}
        for name, counts in columns.items():
            peers = counts.get("peers") or 0
            n = counts.get(measure, 0)
            row[name] = n if measure == "peers" else (
                f"{n} ({n / peers:.1%})" if peers else "0")
        rows.append(row)
    need = cfg["engines"]["correlation"]["profile"]["min_timing_observations"]
    in_chain = (served.matrix["txid"].isin(served.txs).sum()
                if served.matrix is not None else 0)
    relay_rows = 0 if served.matrix is None else len(served.matrix)
    return (
        "The reverse direction (docs/CORRELATION.md): peer -> profile. Each count is "
        "the peers whose profile has that part. `originated claim` means the "
        "origination model named the peer and `eval.origin.flagged_at` did not "
        "withhold it; QUALIFIED and ANNOTATE claims are counted apart, never folded "
        f"into it. A timing signature needs {need} announcements "
        "(`engines.correlation.profile.min_timing_observations`). A cluster link "
        "needs the originated transaction's inputs: the corpus has no chain data, and "
        f"{in_chain} of the served relay matrix's {relay_rows} rows have a txid in the "
        "relay-hop dataset; the rest of the served links come from correlation leads. "
        + ("No capture here carries a version handshake, hence no user agents or "
           "service flags. " if not any(c.get("with a user agent") or c.get(
               "with service flags") for c in columns.values()) else "")
        + "Every profile counted is built from simulated or fixture data and "
        "says so in its header.\n\n" + md_table(pd.DataFrame(rows)) + "\n")


def anomaly_section(fusion: dict, redteam: dict) -> str:
    """State the case against the anomaly engine, with the numbers.

    Not a recommendation to remove it — that is a decision about the
    architecture, and this file's job is to put the number in writing first.
    """
    std, shift = fusion["standard"]["actor"], fusion["shifted"]["actor"]
    example = ""
    misses = redteam.get("misses")
    if misses is not None and len(misses) > 1:
        # Two injected entities with very different anomaly scores and the same
        # fused score is the whole argument in one row pair.
        pair = misses.sort_values("anomaly")
        low, high = pair.iloc[0], pair.iloc[-1]
        if abs(high["anomaly"] - low["anomaly"]) > 0.3:
            example = (
                f"\nThe red-team misses make it concrete. Two injected entities, one "
                f"scoring **{low['anomaly']:.3f}** on anomaly and one **{high['anomaly']:.3f}** "
                f"— a difference of {high['anomaly'] - low['anomaly']:.3f} on the signal — "
                f"both come out of the stacker at **{low['fused']:.3f}** and "
                f"**{high['fused']:.3f}**. The fused score does not move, because nothing "
                "is multiplying it.\n")
    return f"""The stacker fits `anomaly_score`'s coefficient to **{std['coefficients'].get('anomaly_score', 0):.4f}** on the standard set
and **{shift['coefficients'].get('anomaly_score', 0):.4f}** on the shifted set. Not small — zero. The signal is computed,
carried through the pipeline, written into every alert payload, and then
multiplied by nothing.

That is the fitted model's verdict, and it is consistent with the signal's own
discrimination: alone, `anomaly_score` scores **{std['signal_auc'].get('anomaly_score')}** AUC on the standard
set and **{shift['signal_auc'].get('anomaly_score')}** on the shifted one — below 0.5, which is worse than
guessing. The non-negative constraint on the stacker (`fusion.stacker.non_negative`)
forbids it from using a signal by inverting it, so a below-chance signal can only
be given zero weight. Refitting the stack **without** it scores
{std['ablation'].get('anomaly_score', {}).get('stack_without_it')} on the standard set against {std['auc']} with it —
slightly *better* without. Since the coefficient is zero the predictions are
identical either way; the difference is the constrained refit landing on a
different optimum once the column is gone, which is worth knowing but is not
evidence the signal was doing harm.
{example}
**Why it comes out below chance.** IsolationForest finds the population's
outliers, and on this data the outliers are the exchanges: enormous fan-in,
enormous fan-out, thousands of counterparties. Our illicit actors are the
opposite — fresh wallets, few transactions, unremarkable amounts, deliberately
shaped to look ordinary. The engine is working; it is answering a question whose
answer is anti-correlated with the label.

**This is reported, not acted on.** Removing the engine is an architecture
decision with a cost either way: it is four signals instead of five in every
diagram and payload, and an unsupervised detector is the one component that
could in principle flag a typology the rules and the GNN were never shown. Kept
at zero weight it costs {ANOMALY_COST} and misleads anyone reading the alert
payload into thinking it contributed. The number is here so that decision can be
made on it rather than on an impression.
"""


#: Measured in docs/redteam_performance.md — the anomaly refit is the single
#: slowest stage of an incremental re-run.
ANOMALY_COST = "about 2.2 s of every red-team re-run (the slowest stage, see docs/redteam_performance.md)"


ZERO_ATTACK_NOTE = """A false positive here is not the same kind of error as one on the standard set. On
a dataset with crime in it, an alert on a victim or a cash-out wallet is at least
adjacent to something real. On this one there is nothing to be adjacent to: every
alert is the system inventing suspicion from ordinary traffic. Read it against
`alert_precision` on the standard set — the two bound the same quantity from
opposite sides.
"""


CLUSTER_NOTE = """**Read this honestly.** A low ARI does not mean detection is broken — the actor
metrics above are scored through the same clustering and hold up — but it does
bound what the entity-level numbers can mean. Two failure modes pull in opposite
directions:

- **Over-splitting** (many small clusters, a large singleton count) makes an
  actor's wallets land in several entities. Detection still fires on whichever
  entity holds the alerting wallet, so `case_detection_rate` survives, but
  `trace_coverage` has further to walk and the entity an analyst opens shows
  less of the operation than it should.
- **Over-merging** is the dangerous one, and it is what the collapse guard
  exists to catch: one bad change-address guess can fold thousands of unrelated
  users into a single entity, which then alerts as one case. `collapse_guard_fired`
  is how often a cluster exceeded `graph.collapse_guard.max_cluster_wallets` and
  was flagged for review rather than trusted.

The generator's true clusters are per-actor and small; our change-address
heuristics are conservative by design, so over-splitting is expected and
over-merging is not. The numbers say which one is actually happening.
"""


CORRELATION_PREAMBLE = """**Why this is not an AUC delta.** The correlation engine is deliberately absent
from the stacker (`fusion.stacker.signals`), and the reason is not that it
performed badly. An IP link says something about *who* an entity might be, not
about *how risky* it is. Adding it as a fifth signal and reporting the change in
AUC would be asking whether attribution predicts criminality — a question that
is not what the engine is for, and one nobody should want answered in the
affirmative, because a system that treats "we know who you are" as evidence of
guilt is the wrong system.

So it is measured as the claim it actually makes. Each alert carries up to
`fusion.leads_per_entity` leads: *this entity was seen broadcasting from this
address, with this confidence*. The test is how often that is true, and whether
the score attached to it is worth anything — a high-scored lead must be right
more often than a low-scored one, or the number beside it is decoration.

Two truths are reported because they are different questions. **The true
broadcast address** is the actor's own; naming it is attribution. **The observed
address** is where the transaction entered the network — for a masked broadcast
that is a Tor exit or a hosting address, which is a fact about infrastructure
and not about a person. The share of leads whose true address was observable at
all is the ceiling on the first number.
"""


SATURATION_PREAMBLE = """A ranked, explainable alert list is a deliverable of the problem statement, and
a ranking only exists if the scores differ. `docs/demo_script.md` has carried a
line saying every alert scores 1.000 — if that were true the queue would be a
set with a number printed on it, and sorting by risk would do nothing.

Three measurements, because they fail in different ways. The **deciles** show
whether the distribution is spread or spiked. The **count at exactly 1.000** is
the specific claim. The **distinct values in the top 50** is the one an analyst
feels: fifty alerts sharing three scores cannot be worked in order, however well
spread the tail beneath them is. Values are counted at three decimals, because
that is what the console prints — two alerts differing in the fourth are one
value to a reader.

Nothing here is tuned. This is what the fitted stacker produces.

**What was done about it.** The composite is unchanged — no retraining, no
recalibration, no threshold moved. What changed is that the queue no longer
sorts on it alone. `fusion/ordering.py` defines a fixed, published sequence of
tiebreakers, all of them signals already computed, applied in this order:

| # | key | direction | why here |
| --- | --- | --- | --- |
| 1 | `risk_score` | desc | the composite still decides |
| 2 | `rule_typologies` | desc | distinct rule detectors that fired — two agreeing independently is a stronger case than one firing twice, and it is the only tiebreaker counting *separate* evidence |
| 3 | `taint_hops` | asc | hops from a watchlist seed; one hop is more urgent than four. No path sorts last, not first — absence is not proximity zero |
| 4 | `lead_confidence` | desc | the strongest attribution lead: of two equal alerts, open the one an ISP request could act on |
| 5 | `tx_count` | desc | the entity's transaction volume — a bigger operation, all else equal |
| 6 | `entity_id` | asc | never a judgement, only a guarantee: with this last the order is total, so the same dataset gives the same queue on every run |

The four tiebreak columns and the composed key travel on the alert record, so
the console, the API and the PDF order identically rather than each re-deriving
the rule; `tests/test_ordering.py` asserts that the exported key reproduces the
queue and that the order is unchanged under every rotation of the input.

This is an ordering fix, not a scoring fix. It makes the queue legible and
stable; it does not make the composite discriminate, and the numbers below say
how far it gets.
"""


DIAGNOSIS_PREAMBLE = """There are only two places the composite can lose resolution, and the fix differs
for each.

* **The sigmoid.** The stacker is a logistic regression: the score is
  `1 / (1 + exp(-z))` over a linear `z`. Past about z = 7.6 that curve is flat
  to the three decimals the console prints. Inputs that differ perfectly well in
  log-odds then arrive at the same displayed score — the information exists in
  `z` and the display throws it away.
* **The inputs.** If entities genuinely have identical signal vectors, no
  transformation of `z` can separate them. That is a detector problem, not a
  presentation one.

The measurement that distinguishes them is how many alerts share an *identical
input vector*. Different inputs and the same score is the sigmoid; the same
inputs were never distinguishable.
"""


DIAGNOSIS_VERDICT = """**The verdict: it is the sigmoid — but fixing it recovers less than it looks.**

Almost every alert has its own input vector ({vectors} of {alerts}), so the
entities are not indistinguishable; the logistic function is flattening them.
The log-odds spread is {spread}, and every bit of it above {flat} prints as
1.000.

The catch is the second number. The logit itself has only **{logits} distinct
values** across {alerts} alerts, and that is not the sigmoid's fault. Of the
four signals, `gnn_score` is absent, `anomaly_score` carries
{anomaly_distinct} distinct values and a coefficient of zero (§8), and the two
signals that actually have weight are coarse: {coarse}. The composite cannot
have more resolution than its weighted inputs.

So a calibration change — ranking on `z`, or any monotone rescale of it — would
take the queue from {displayed} displayed values to **at most {logits}**, not to
{alerts}. It is a real improvement and a cheap one, and it is not a ranking.

**What it would cost.** Ranking on the logit is monotone, so every ordering and
every AUC in this report is unchanged by construction. What changes is the
number on screen: `alert_threshold` is {threshold} on a probability scale and would
have to be re-derived on whatever scale replaced it, which means the threshold is
re-registered rather than tuned — a protocol change, not a code change. Every
score quoted anywhere would move, so this file and the README would need
regenerating together.

The alternative, retraining with stronger regularisation so the coefficients
stop diverging, is the textbook fix for a near-separable fit — coefficients on this
data reach three figures — but it is a retrain, it changes AUC, and it needs
its own validation pass. Neither was done here.

**The real ceiling is upstream.** {logits} levels is what four signals give when
two are silent and two are coarse. Finer resolution has to come from the rules
engine emitting a continuous confidence rather than a handful of bands, or from
a signal that actually varies — which is the same conversation as §8.
"""


NON_ACTOR_NOTE = """**Why the headline excludes two of the five.** `coinjoin` and
`same_actor_cluster` are pre-registered in `docs/detection_unit_protocol.md` as
**not actors** — CoinJoin is mixing, which is suspicious but not by itself a
crime, and `same_actor_cluster` is a test of the clustering rather than an
offence. That document was written and committed before any of these injections
were run, so the exclusion is a rule applied, not a choice made after seeing
which typologies scored badly.

Counting them as misses would score the system for declining to alert on
something it is right not to alert on, and would make "flag every CoinJoin" look
like an improvement. It is not one: it is a false-positive generator pointed at
a legal privacy tool.
"""


REDTEAM_PREAMBLE = """`eval.redteam_runs` injections driven through `api.redteam.execute` — the same
function `POST /redteam/runs` calls, not a re-implementation of it. Every
typology appears equally often, with hops, amounts, wallet counts and broadcast
routes spread across the ranges the form exposes, so the rate is not an average
over one corner of the parameter space. The batch runs against a **copy** of the
shifted set: injection appends to `transactions.csv` and `ground_truth.json`,
and an evaluation that consumed the canonical dataset would make every other
number in this file irreproducible.

State is threaded from one injection to the next, as it is on a live server —
so this measures a system whose dataset is growing under it, which is the
condition the demo runs in.
"""


WORSE = """**What got worse, and why.**

- **`wallet_recall_broad` is {recall}, and it is now a secondary number.** Under
  the broad label most illicit wallets are single-transaction layering hops. The
  system alerts on the operations, not on every hop, so this number is
  structurally low and always will be. It is reported for comparability with the
  previous pass, not as a target.
- **Stacker AUC on the actor label is {actor_auc} (standard) / {actor_auc_shifted}
  (shifted), against {broad_auc} on the old broad label.** The actor label is
  harder: it excludes the cash-out and victim wallets that sat next to the easy
  rule alerts. Same model, same non-negative constraint, a label that no longer
  hands out credit for adjacency.
- **The actor denominator is tiny: {actors} illicit operations on the standard
  set, {actors_shifted} on the shifted set.** The canonical dataset was sized for
  wallet-level statistics. A case detection rate over {actors} cases moves in
  20-point steps and a single miss would dominate it; seed B is reported below for
  exactly this reason. Fixing it properly means a larger canonical corpus, which
  would invalidate every pre-registered origin number in the same file — so it is
  named here rather than quietly done.
- **Turning the rank penalty off entirely scores {off_top1} top-1 but a
  cost-weighted score of 0.000**: with no penalty the top candidate is a public
  relay often enough that `low_confidence_origin` abstains on all {n} estimates.
  Accuracy alone would have called this the best configuration. It is the least
  useful one.
- **The split filter is 4.5pp more accurate than the old combined filter
  ({split_top1} vs {combined_top1}) and abstains far more often** ({abstained_split}
  of {n}). That is the trade it was built to make: it names an uninvolved third
  party {wrong_split} times where the combined filter did so {wrong_combined} times.
- **`low_confidence_origin`'s precision as a predictor of literal absence is
  {flag_precision} on seed B.** It was never a good predictor of that; the rename
  exists because the old name claimed it was.
"""


CLOSING = """
## 14. Decisions taken in this pass

**The unit of detection is the actor.** Pre-registered in
`docs/detection_unit_protocol.md` before the label was built or the stacker
retrained. An actor is one ground-truth illicit operation: the ransomware
collector with its peel-chain change addresses, or one layering instance's
source, hops and sink grouped back together. Victims and cash-out counterparties
are not part of the actor. Wallet-level recall is kept as a secondary number so
this pass stays comparable with the last one.

**The origin filter is split.** `known_bitcoin_relay` keeps its rank penalty: a
Bitnodes-listed relay forwards other people's traffic and is never a plausible
sender. `tor_exit` and `hosting_vpn` lose it: when a broadcast is masked, that
address *is* where the transaction entered the network, and penalising it in the
ranking pushed the estimator off the right answer for no gain. Those candidates
are now reported as **anonymized entry points**, and only the attribution
confidence handed to the correlation engine is reduced
(`attribution_confidence_factor`). The cost-weighted score is what settled it —
under plain accuracy the honest comparison was unavailable, because accuracy
prices a confident wrong attribution the same as a miss.

**`origin_likely_unobserved` is now `low_confidence_origin`** in the API, in
alert payloads, in config and in the docs. The old name asserted that the true
origin was absent from the data, which nothing inside the estimator can know;
measured as a predictor of that event its precision is ~0.2. What it actually
separates is weak estimates from strong ones, and it is now named for that. Its
cutoff is chosen on seed A by the pre-registered rule and reported unchanged on
seed B.

**Correlation stays out of the risk score.** Unchanged from the previous pass:
an IP correlation says something about *who*, not about whether an entity is
risky. It is surfaced per alert as attribution leads, each now labelled
`anonymized entry point` when the candidate is a Tor exit or hosting address.
"""


def ground_truth_section(add, result: dict, cfg: dict, origination: str = "") -> None:
    """Section 9. Every other section's origin numbers describe our simulator;
    this one is the harness for measuring the same thing on real relay data.

    The first line states the data source, because a simulated run and a signet
    run produce the same table shape and must never be confused for one another.
    """
    signet = result["signet"]
    add(f"\n**Data source: {result['data_source']}.** "
        + ("A signet capture is present and scored below.\n" if signet else
           "**No signet capture is present in this run**, so every signet row below is "
           "marked PENDING and the only measured rows come from the gossip simulation. "
           "A simulated row is never a statement about Bitcoin.\n"))
    add("\nProtocol pre-registered in `docs/GROUND_TRUTH.md`, committed with the harness "
        "and\nbefore any signet number existed. Capture setup, and how each topology "
        "condition is\nforced and verified, are in the same document.\n")
    add("\nThe two topology conditions are **separate measurements and are never "
        "pooled**:\n\n"
        "* **adjacent** — the broadcaster is directly peered with the observer. The "
        "trivial\n  upper bound: the first announcement we see really is the source's "
        "own.\n"
        "* **non_adjacent** — at least one hop between them. The real result, and the "
        "only\n  one that says anything about a deployment.\n")

    if result["pending"]:
        add(f"\n**PENDING: {', '.join(result['pending'])}.** "
            "No sealed signet bundle for "
            + ("either condition" if len(result["pending"]) > 1 else "this condition")
            + " is present. The rows are left in place rather than filled from the "
            "simulation.\n")
    if result.get("skipped_bundles"):
        add("\nBundles present but not scored as signet (their label file does not say "
            "`source: signet`): "
            + ", ".join(f"`{b['bundle']}` ({b['source']})" for b in result["skipped_bundles"])
            + ". Test fixtures live in the same directory and are excluded by that rule.\n")

    for condition in ("adjacent", "non_adjacent"):
        entry = signet.get(condition)
        add(f"\n### {condition} — signet\n")
        if entry is None:
            add("PENDING — no sealed capture for this condition.\n")
            continue
        if entry.get("status") != "scored":
            add(f"**{entry['status']}**\n")
            for failure in entry.get("failures", []):
                add(f"\n* {failure}\n")
            continue
        add(f"Bundle `{entry['bundle']}`, manifest verified, "
            f"{entry['transactions_scored']} transactions scored.\n\n")
        add(md_table(entry["table"]))
        add("\n" + md_table(pd.DataFrame([entry["noise_floor"]])))
        add("\n" + md_table(pd.DataFrame([entry["wtxid_resolution"]])))

    simulated = result["simulated"]
    add("\n### simulated — `generator/`'s 500-node gossip network\n")
    add(f"`condition=\"simulated\"`. {simulated['describes']}. Deterministic, always "
        "available, and **not a stand-in for a signet run**: the simulation is observed "
        "at many relays, so its trees carry the positional structure a single-observer "
        "capture does not have.\n\n")
    add(md_table(simulated["table"]))
    add("\nRow 1 is the floor — earliest sighting wins, no class weighting, no "
        "abstention. It is\nwhat naive analysis does, and under the pre-registered cost "
        "weights it scores\n**negative**: naming the wrong uninvolved address is priced "
        "at -3, and the floor does it\noften. Rows 2-4 are the three estimators from "
        "`engines/propagation/` unchanged. Row 5\nis the supervised origination model, "
        "which cannot be scored here: it reads one observer's\nrelay matrix, and hop "
        "records sampled at many relays do not make one. It is measured on\nthe capture "
        "corpus at the end of this section, beside these four baselines on one split.\n")
    add("\n**Relay-delay noise floor.** How far apart announcements of the same "
        "transaction\nactually arrive, and what that implies for any timing-based "
        "estimator:\n\n")
    add(md_table(pd.DataFrame([simulated["noise_floor"]]), floats=6))
    add("\n`timing ceiling` is the share of multi-peer transactions where the true "
        "origin\nannounced *first and by more than the clock resolution*. No estimator "
        "that reads only\ntiming can exceed it, however it weights what it reads.\n")
    matrix = result.get("relay_features")
    if matrix is not None and len(matrix):
        add("\n### The relay feature matrix\n")
        add("`features/relay.py`, grain `(txid, peer_ip, capture_id)` — the input the "
            "supervised\norigination model reads. Every column, its null semantics "
            "and its causality\nargument are in `docs/FEATURE_SCHEMA_RELAY.md`. The "
            "`source` column is carried here\nbecause a fixture-derived row must never "
            "be read as a signet one.\n\n")
        add(md_table(matrix, floats=4))
        add("\n`degenerate` is the share of rows whose transaction had 0 or 1 candidate "
            "— nothing to\nrank. `scope_out` is the share where no candidate could "
            "plausibly be the sender, which\nis the zero-ceiling case in feature form. "
            "Both are rows a model must abstain on rather\nthan learn from, which is "
            "why they are counted here and why the model below never scores them.\n`quarantined` counts "
            "announcements still identified by a wtxid, kept in a separate file and "
            "never\nmerged into the matrix.\n")

    add("\n**One observer sees a star, not a tree.** A single-observer capture yields "
        "one edge\nper announcement — peer to observer — so rumor centrality, which "
        "maximises over tree\nposition, has nothing to rank on once the observer is "
        "excluded as a candidate for its\nown observations. That is not a defect in "
        "Shah & Zaman; it is what their estimator\ndoes when the observed topology "
        "carries no positional information. On the signet\nconditions only timing "
        "carries signal, which is why the noise floor above is the\nnumber that bounds "
        "them. A multi-observer capture would restore the topology.\n")
    add(origination)


def main(argv=None) -> None:
    cfg = config.load()
    ap = argparse.ArgumentParser(prog="eval.report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=cfg["eval"]["results_path"])
    ap.add_argument("--rebuild", action="store_true", help="regenerate the datasets")
    args = ap.parse_args(argv)
    report = build_report(cfg, args.rebuild)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(f"wrote {path} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
