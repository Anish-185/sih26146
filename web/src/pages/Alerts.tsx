/** The alert queue: triage, verdict, next.
 *
 *  Filters and sorting live in the URL so a queue state can be pasted to a
 *  colleague, and the verdict a reader gives is sent immediately — an
 *  investigator who marks something and closes the tab has still marked it. */
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { Alert } from "../api/types";
import { useApi } from "../lib/useApi";
import { useEnter } from "../lib/motion";
import { patternLabel } from "../lib/format";
import {
  AlertTable,
  filterAlerts,
  sortAlerts,
  type SortKey,
  type Verdict,
} from "../components/AlertTable";
import { ActorQueue, ACTOR_STATEMENT } from "../components/ActorQueue";
import { ErrorNote, Label, Notice, SkeletonRows } from "../components/ui";
import { Shell } from "../components/Shell";
import { useToast } from "../components/Toasts";

const ANCHORS = [
  { id: "queue", label: "Queue" },
  { id: "how", label: "How to read this" },
];

export function Alerts() {
  const [params, setParams] = useSearchParams();
  const toast = useToast();
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [pending, setPending] = useState<Record<string, boolean>>({});

  const minScore = Number(params.get("min_score") ?? 0);
  const patternType = params.get("pattern_type") ?? "";
  const sortKey = (params.get("sort") as SortKey) ?? "risk_score";
  const sortDir = (params.get("dir") as "asc" | "desc") ?? "desc";
  // Actors by default; ?view=entities is the queue as it was before actors.
  const view = params.get("view") === "entities" ? "entities" : "actors";
  const actors = useApi((signal) => api.actors(signal), []);

  const { data, error, loading } = useApi(
    (signal) => api.alerts({ limit: 200 }, signal),
    [],
  );

  const all = useMemo(() => data?.alerts ?? [], [data]);
  const patterns = useMemo(
    () => [...new Set(all.flatMap((alert) => alert.pattern_types))].sort(),
    [all],
  );
  const rows = useMemo(
    () => sortAlerts(filterAlerts(all, { minScore, patternType }), sortKey, sortDir),
    [all, minScore, patternType, sortKey, sortDir],
  );

  // When the fused score saturates, every row reads the same number and the
  // queue cannot be ranked by it. Say so rather than letting a reader believe
  // 197 identical scores mean 197 equally urgent cases.
  const spread =
    rows.length > 1
      ? Math.max(...rows.map((a) => a.risk_score)) - Math.min(...rows.map((a) => a.risk_score))
      : 1;
  const saturated = rows.length > 1 && spread < 0.005;

  const head = useEnter<HTMLElement>(0);
  const table = useEnter<HTMLElement>(1);

  const update = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  };

  const onSort = (key: SortKey) => {
    const dir = sortKey === key && sortDir === "desc" ? "asc" : "desc";
    const next = new URLSearchParams(params);
    next.set("sort", key);
    next.set("dir", dir);
    setParams(next, { replace: true });
  };

  const onVerdict = async (alert: Alert, verdict: Verdict) => {
    setPending((p) => ({ ...p, [alert.alert_id]: true }));
    try {
      await api.feedback(alert.alert_id, verdict);
      setVerdicts((v) => ({ ...v, [alert.alert_id]: verdict }));
      toast(
        verdict === "confirmed"
          ? `Confirmed ${alert.entity_id}`
          : `Marked ${alert.entity_id} a false positive`,
      );
    } catch (e) {
      toast(`Could not record the verdict: ${(e as Error).message}`);
    } finally {
      setPending((p) => ({ ...p, [alert.alert_id]: false }));
    }
  };

  return (
    <Shell anchors={ANCHORS}>
      <section className="section" id="queue" ref={head}>
        <div className="section-head">
          <h1>{view === "actors" ? "Actor queue" : "Alert queue"}</h1>
          <span className="label">
            {view === "actors"
              ? actors.loading
                ? "loading"
                : `${actors.data?.total ?? 0} actors`
              : loading
                ? "loading"
                : `${rows.length} of ${all.length} alerts`}
          </span>
        </div>

        <div className="filters" role="group" aria-label="Queue view">
          <button
            type="button"
            className={`btn${view === "actors" ? "" : " btn-quiet"}`}
            aria-pressed={view === "actors"}
            onClick={() => update("view", "")}
          >
            actors
          </button>
          <button
            type="button"
            className={`btn${view === "entities" ? "" : " btn-quiet"}`}
            aria-pressed={view === "entities"}
            onClick={() => update("view", "entities")}
          >
            entities (previous view)
          </button>
        </div>

        {view === "entities" && (
        <div className="filters">
          <label className="field">
            <span className="label">Min risk</span>
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.05}
              value={minScore}
              onChange={(event) => update("min_score", event.target.value)}
              aria-label="Minimum risk score"
            />
            <span className="num">{minScore.toFixed(2)}</span>
          </label>

          <label className="field">
            <span className="label">Pattern</span>
            <select
              value={patternType}
              onChange={(event) => update("pattern_type", event.target.value)}
              aria-label="Pattern type"
            >
              <option value="">all</option>
              {patterns.map((pattern) => (
                <option key={pattern} value={pattern}>
                  {patternLabel(pattern)}
                </option>
              ))}
            </select>
          </label>

          {(minScore > 0 || patternType) && (
            <button
              type="button"
              className="btn btn-quiet"
              onClick={() => setParams(new URLSearchParams(), { replace: true })}
            >
              clear filters
            </button>
          )}
        </div>
        )}
      </section>

      {view === "entities" && saturated && (
        <div style={{ marginBottom: "var(--sp-4)" }}>
          <Notice title="Risk scores are saturated">
            Every alert here scores {rows[0].risk_score.toFixed(3)}. The fused score separates
            alerts from everything else, but it cannot rank these against each other — order
            this queue by pattern or wallet count, and read the reasons.
          </Notice>
        </div>
      )}

      <section ref={table} style={{ marginTop: "var(--sp-2)" }}>
        {view === "actors" ? (
          actors.error ? (
            <ErrorNote error={actors.error} />
          ) : actors.loading ? (
            <SkeletonRows rows={8} />
          ) : (actors.data?.actors.length ?? 0) === 0 ? (
            <p className="soft">No actor is above the alert threshold.</p>
          ) : (
            <>
              <p className="soft">{actors.data?.statement ?? ACTOR_STATEMENT}</p>
              <ActorQueue actors={actors.data!.actors} />
            </>
          )
        ) : error ? (
          <ErrorNote error={error} />
        ) : loading ? (
          <SkeletonRows rows={8} />
        ) : rows.length === 0 ? (
          <p className="soft">
            Nothing matches these filters. Lower the minimum risk, or clear the pattern.
          </p>
        ) : (
          <AlertTable
            alerts={rows}
            verdicts={verdicts}
            pending={pending}
            onVerdict={onVerdict}
            sort={{ key: sortKey, dir: sortDir }}
            onSort={onSort}
          />
        )}
      </section>

      <section className="section" id="how">
        <div className="section-head">
          <h2>How to read this</h2>
          <span className="label">before you act on a row</span>
        </div>
        <div className="measure stack" style={{ gap: "var(--sp-3)" }}>
          <p className="soft">
            The risk score is a fused output of four engines — rules, anomaly, graph neural
            network and taint — fitted on synthetic labels. It ranks work; it does not prove
            anything. Open a case and read the reason and the evidence before deciding.
          </p>
          <p className="soft">
            <Label>Verdicts</Label>
            Confirm and reject are recorded for recalibration. They do not change this score
            or remove the entity: a rejected alert stays visible, dimmed, so the queue keeps
            showing what was looked at.
          </p>
        </div>
      </section>
    </Shell>
  );
}
