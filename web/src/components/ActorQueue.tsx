/** The actor queue: address clusters joined to the peer identities the
 *  evidence links them to (fusion/actors.py). An actor's name is a handle for
 *  that group; it never implies a person or an organisation. */
import { Link } from "react-router-dom";
import type { Actor } from "../api/types";
import { formatId } from "../lib/format";
import { Chip, RiskChip } from "./ui";

export const ACTOR_STATEMENT =
  "An actor is a group of addresses and network identities linked by evidence; it does not name or imply a person or an organisation.";

export function ActorQueue({ actors }: { actors: Actor[] }) {
  return (
    <div className="table-wrap">
    <table className="data" aria-label="Actor queue">
      <thead>
        <tr>
          <th>Actor</th>
          <th className="num">Risk</th>
          <th className="num">Membership confidence</th>
          <th>Member clusters</th>
          <th>Linked peers</th>
          <th>Origin tiers</th>
        </tr>
      </thead>
      <tbody>
        {actors.map((a) => (
          <tr key={a.actor_id}>
            <td>
              <Link to={`/actors/${encodeURIComponent(a.actor_id)}`}>{a.name}</Link>
            </td>
            <td className="num">
              {a.risk_score.toFixed(3)} <RiskChip score={a.risk_score} />
            </td>
            <td className="num">{a.membership_confidence.toFixed(2)}</td>
            <td>
              {a.members.slice(0, 3).map((m) => (
                <Link key={m} className="mono" to={`/entities/${encodeURIComponent(m)}`}>
                  {formatId(m)}{" "}
                </Link>
              ))}
              {a.members.length > 3 && <span className="soft">+{a.members.length - 3}</span>}
            </td>
            <td>
              {a.links.slice(0, 3).map((l) => (
                <Chip key={`${l.peer}-${l.cluster_id}`} title={l.evidence_chain}>
                  <Link to={`/peers/${encodeURIComponent(l.peer)}`}>{l.peer}</Link> · {l.basis} ·{" "}
                  {l.confidence.toFixed(2)}
                  {l.joins ? "" : " · attached, not joining"}
                </Chip>
              ))}
              {a.links.length === 0 && <span className="soft">none</span>}
              {a.links.length > 3 && <span className="soft"> +{a.links.length - 3}</span>}
            </td>
            <td>{a.origin_tiers.length ? a.origin_tiers.join(", ") : <span className="soft">—</span>}</td>
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  );
}
