/** One actor: its member clusters, the peers linked to them with basis and
 *  confidence, the validity tiers of any origin claims, and the way down to
 *  the per-entity, per-peer and per-transaction views. Opening this page is
 *  recorded in the custody ledger by the server. */
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { ACTOR_STATEMENT } from "../components/ActorQueue";
import { Shell } from "../components/Shell";
import { useToast } from "../components/Toasts";
import { ErrorNote, Label, RiskChip, SkeletonRows } from "../components/ui";
import { formatId } from "../lib/format";
import { useApi } from "../lib/useApi";

export function Actor() {
  const { id = "" } = useParams();
  const toast = useToast();
  const [verdict, setVerdict] = useState<string | null>(null);
  const { data, error, loading } = useApi((signal) => api.actor(id, signal), [id]);

  const record = async (status: "confirmed" | "false_positive") => {
    try {
      await api.actorVerdict(id, status);
      setVerdict(status);
      toast(`Recorded ${status.replace("_", " ")} for actor ${id}`);
    } catch (e) {
      toast(`Could not record the verdict: ${(e as Error).message}`);
    }
  };

  return (
    <Shell>
      <section className="section">
        {error ? (
          <ErrorNote error={error} />
        ) : loading || !data ? (
          <SkeletonRows rows={6} />
        ) : (
          <>
            <div className="section-head">
              <h1>{data.name}</h1>
              <span className="label">
                risk {data.risk_score.toFixed(3)} · membership confidence{" "}
                {data.membership_confidence.toFixed(2)}
              </span>
            </div>
            <p className="soft">{data.statement ?? ACTOR_STATEMENT}</p>
            <p>
              <RiskChip score={data.risk_score} />{" "}
              <span className="soft" style={{ fontSize: "var(--fs-small)" }}>
                {data.confidence_basis}
              </span>
            </p>
            <p>
              <button type="button" className="btn" onClick={() => record("confirmed")}>
                confirm
              </button>{" "}
              <button type="button" className="btn btn-quiet" onClick={() => record("false_positive")}>
                false positive
              </button>
              {verdict && <span className="soft"> recorded: {verdict.replace("_", " ")}</span>}
            </p>

            <h2>Member clusters</h2>
            <ul>
              {data.member_detail.map((m) => (
                <li key={m.entity_id}>
                  <Link className="mono" to={`/entities/${encodeURIComponent(m.entity_id)}`}>
                    {formatId(m.entity_id)}
                  </Link>{" "}
                  <span className="soft">
                    entity risk {m.entity_risk.toFixed(3)} · weight {m.weight.toFixed(2)}
                    {m.entity_id === data.anchor ? " · anchor" : ""}
                    {m.alerted ? " · alerted" : ""}
                  </span>
                </li>
              ))}
            </ul>

            <h2>Linked peers</h2>
            {data.links.length === 0 ? (
              <p className="soft">No peer identity is linked to these clusters.</p>
            ) : (
              <ul>
                {data.links.map((l) => (
                  <li key={`${l.peer}-${l.cluster_id}`}>
                    <Link to={`/peers/${encodeURIComponent(l.peer)}`}>{l.peer}</Link> ({l.peer_kind}
                    {l.ip_class ? `, ${l.ip_class}` : ""}) → cluster{" "}
                    <span className="mono">{formatId(l.cluster_id)}</span> by {l.basis},{" "}
                    confidence {l.confidence.toFixed(2)}
                    {l.joins ? "" : " — attached, not joining"}
                    {l.origin_tiers.length > 0 && ` · origin tiers ${l.origin_tiers.join(", ")}`}
                    <div className="soft" style={{ fontSize: "var(--fs-small)" }}>
                      <Label>{l.evidence_chain}</Label>
                      {l.evidence.map((e) => (
                        <Link key={`${e.basis}-${e.txid}`} className="mono" to={`/tx/${e.txid}`}>
                          {formatId(e.txid)}{" "}
                        </Link>
                      ))}
                      {l.evidence_total > l.evidence.length &&
                        ` (+${l.evidence_total - l.evidence.length} more)`}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </section>
    </Shell>
  );
}
