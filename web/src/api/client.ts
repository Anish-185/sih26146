/** Every call the console makes. One place, so the base URL and the error
 *  shape are decided once.
 *
 *  VITE_API_BASE defaults to same-origin, which is what FastAPI serving
 *  web/dist gives us; the dev server proxies instead (see vite.config.ts). */
import type {
  ActorDetail,
  ActorsPage,
  AlertsPage,
  AsnProfile,
  EntityDetail,
  EntityGraph,
  PeerProfile,
  Propagation,
  Stats,
  TxFingerprint,
  TxOrigination,
} from "./types";

export const API_BASE = (import.meta.env?.VITE_API_BASE ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    throw new ApiError(await detail(response), response.status);
  }
  return (await response.json()) as T;
}

async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return typeof body?.detail === "string" ? body.detail : response.statusText;
  } catch {
    return response.statusText;
  }
}

export interface AlertQuery {
  limit?: number;
  offset?: number;
  minScore?: number;
  entityType?: string;
  patternType?: string;
}

export function alertsPath(query: AlertQuery = {}): string {
  const params = new URLSearchParams();
  if (query.limit != null) params.set("limit", String(query.limit));
  if (query.offset) params.set("offset", String(query.offset));
  if (query.minScore) params.set("min_score", String(query.minScore));
  if (query.entityType) params.set("entity_type", query.entityType);
  if (query.patternType) params.set("pattern_type", query.patternType);
  const qs = params.toString();
  return `/alerts${qs ? `?${qs}` : ""}`;
}

export const api = {
  stats: (signal?: AbortSignal) => get<Stats>("/stats", signal),
  alerts: (query: AlertQuery = {}, signal?: AbortSignal) =>
    get<AlertsPage>(alertsPath(query), signal),
  entity: (id: string, signal?: AbortSignal) =>
    get<EntityDetail>(`/entities/${encodeURIComponent(id)}`, signal),
  entityGraph: (id: string, hops: number, signal?: AbortSignal) =>
    get<EntityGraph>(`/entities/${encodeURIComponent(id)}/graph?hops=${hops}`, signal),
  propagation: (txid: string, signal?: AbortSignal) =>
    get<Propagation>(`/transactions/${encodeURIComponent(txid)}/propagation`, signal),
  fingerprint: (txid: string, signal?: AbortSignal) =>
    get<TxFingerprint>(`/transactions/${encodeURIComponent(txid)}/fingerprint`, signal),
  origination: (txid: string, signal?: AbortSignal) =>
    get<TxOrigination>(`/transactions/${encodeURIComponent(txid)}/origination`, signal),
  /** Every call is recorded in the custody ledger by the server. */
  peerProfile: (peer: string, signal?: AbortSignal) =>
    get<PeerProfile>(`/peers/${encodeURIComponent(peer)}/profile`, signal),
  asnProfile: (asn: string, signal?: AbortSignal) =>
    get<AsnProfile>(`/asns/${encodeURIComponent(asn)}/profile`, signal),

  actors: (signal?: AbortSignal) => get<ActorsPage>("/actors?limit=200", signal),
  /** Every call is recorded in the custody ledger by the server. */
  actor: (id: string, signal?: AbortSignal) =>
    get<ActorDetail>(`/actors/${encodeURIComponent(id)}`, signal),

  reportUrl: (id: string) => `${API_BASE}/entities/${encodeURIComponent(id)}/report`,

  async feedback(alertId: string, status: "confirmed" | "false_positive") {
    const response = await fetch(
      `${API_BASE}/alerts/${encodeURIComponent(alertId)}/feedback`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ status }),
      },
    );
    if (!response.ok) throw new ApiError(await detail(response), response.status);
    return (await response.json()) as { alert_id: string; status: string; recorded: number };
  },

  async actorVerdict(actorId: string, status: "confirmed" | "false_positive") {
    const response = await fetch(`${API_BASE}/actors/${encodeURIComponent(actorId)}/verdict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!response.ok) throw new ApiError(await detail(response), response.status);
    return (await response.json()) as { actor_id: string; status: string; recorded: number };
  },
};
