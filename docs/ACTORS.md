# Actors

An **actor** is one or more address clusters (`graph/`) joined to zero or more
peer identities (the P7 peer profiles, `engines/correlation/profile.py`) by the
links the evidence supports. It is what `fusion/actors.py` builds. The console
shows the actor queue by default.

**An actor is a handle, not a party.** Actors are named `actor A-17`. The name
groups addresses and network identities that the evidence links. It never
names or implies a person or an organisation. Every API response and console
view carries that statement.

## What was there before

Alerts were already entity-level. `fusion/pipeline.py` scores each entity (an
address cluster, or a lone wallet) with the fitted stacker, and the alert
carries its IP leads from `engines/correlation`. So this phase joins in the
network side. It does not rebuild scoring.

## Links

A link is a (peer, cluster) pair from `profile._linked_clusters`. It carries:

* **basis**: `origination` (the origination model named the peer as origin of
  a transaction spending the cluster's inputs), `correlation lead` (the
  forward engine's (cluster, IP) association), or `both`;
* **confidence**: origination is `1 - exp(-Σ calibrated probability / k)`. A
  correlation lead is an evidence-weighted count and is **not calibrated**.
  With both, the stronger of the two is used; they read overlapping evidence,
  so they are not combined;
* **validity tiers** of the origin claims behind it;
* **evidence**: the raw rows (relay-matrix rows or relay-hop rows), the
  transaction, and the input addresses. The chain is
  raw rows → transaction → input addresses → cluster.

A **QUALIFIED** origination joins only with its restricted meaning:

* a **COINJOIN**-qualified claim names the broadcasting peer only. It never
  links that peer to any participant's cluster, so no link is created for it
  (`profile._linked_clusters`). `actors.build` also refuses any link whose
  origination evidence is COINJOIN-qualified;
* a **TOR_ONION**-qualified claim links an *onion identity*, never an IP.

Fingerprints contribute nothing to membership.

## Joining (fixed before any evaluation)

Every link is kept and shown. Only a link that `may_join` puts two clusters
into one actor. Two conditions must both hold:

1. the peer is a residential/unknown IP or an onion identity. A hosting/VPN
   range, a Tor exit or a public relay serves many unrelated users by
   definition, as the validity layer already treats it;
2. an origination claim stands behind the link (basis `origination` or
   `both`), not a correlation lead alone.

Any other link **attaches** its peer to the actor of the cluster it names. It
is listed with its basis and confidence, it lowers the actor's membership
confidence, and it merges nothing. The first draft joined through every link.
On the demo data it chained 357 clusters and 217 peers into one component,
through shared infrastructure. This rule was written to prevent that before
the ground-truth evaluation was run. It was not tuned on the evaluation.

## Scoring: one path

An actor is scored by **the same fitted stacker** as an entity, on the same
five signals. For each signal, the actor's value is the largest member value,
weighted by how sure the join to that member is:

* the **anchor** (the member the stacker scores highest) counts in full;
* every other member counts by the weakest link on the strongest path from the
  anchor (the maximin spanning tree over joining links).

An actor of one cluster therefore scores exactly as its entity does. An
uncertain join lowers what a member can add; it is not silently dropped. An
actor alerts at the same `fusion.alert_threshold`.

**Membership confidence** is the weakest link among all the actor's links,
joining or attached. Its basis is stated with it, and part of it is
uncalibrated (the correlation leads).

## Surfaces

* `python -m fusion.pipeline` writes `data/processed/final_actors.json` beside
  the alerts, built from the same bundle and stacker, and seals it in the
  custody entry for the analysis.
* API:
  * `GET /actors` lists the actor queue; `alerted_only=false` lists every actor.
  * `GET /actors/{id}` returns one actor with score, membership confidence,
    member clusters, linked peers with basis, validity tiers and evidence, and
    `drill_down` links to `/entities/…`, `/peers/…/profile` and the
    transactions. Every view is recorded in the custody ledger
    (`actor.view`), found or not.
  * `POST /actors/{id}/verdict` records `confirmed` or `false_positive` in
    `actor_feedback.parquet`. That file is kept apart from the entity feedback
    the stacker trains on, and every verdict is recorded in the ledger
    (`actor.verdict`).
* Console: the queue page shows actors by default and has a toggle to the
  entity queue (`?view=entities`). `/actors/:id` is the actor page.

## Evaluation

Section 13 of `eval/results.md`: the actor queue against the entity queue on
the standard and shifted generator datasets. It reports:

* the alert count;
* precision@k and recall@k for illicit operations (`eval.actors.actors_of`);
* items reviewed before finding 1, 3 and all operations;
* actor purity and the wrong-merge rate.

It is simulated and cross-topology: the generator's network configuration is
not among the origination corpus's. The same section also scores the
origination model on the served demo capture, and attributes the fingerprint
novelty flags on the demo to the generator differences that cause them.

## Limits

* Generator datasets hold a handful of illicit operations (4–5), so
  precision@k saturates for both queues and the triage comparison has little
  power.
* Joins rest on origination claims. On data where the model abstains, actors
  are mostly single clusters.
* Correlation-lead confidences are not calibrated, so membership confidence is
  an ordering, not a probability.
