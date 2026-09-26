/** The shapes api/app.py actually returns. Kept narrow on purpose: fields the
 *  console does not draw are not typed here. */

export type IpClass =
  | "residential_or_unknown"
  | "tor_exit"
  | "hosting_vpn"
  | "known_bitcoin_relay";

/** analysis/validity.py's verdict. ABSTAIN withholds the answer, QUALIFIED
 *  restricts what it may claim, ANNOTATE only flags it. */
export interface Validity {
  tier: "PASS" | "ABSTAIN" | "QUALIFIED" | "ANNOTATE" | "NOT_ASSESSED";
  reason: string | null;
  reasons: string[];
  confidence: number | null;
  evidence: string[];
}

/** What an origin answer may claim. Only `ip_attribution` is an IP attribution. */
export interface Answer {
  kind: "ip_attribution" | "onion_identity" | "broadcasting_peer";
  ip?: string;
  onion?: string;
  actionable?: string;
  input_ownership?: string;
}

export interface Lead {
  ip: string;
  ip_class: IpClass;
  confidence: number;
  observations: number;
  anonymized_entry_point?: boolean;
  label?: string;
  evidence: string;
  calibration_basis?: string;
  validity: Validity;
}

export interface Alert {
  alert_id: string;
  entity_id: string;
  entity_type: "cluster" | "wallet";
  pattern_types: string[];
  risk_score: number;
  reason: string;
  evidence: string[];
  top_signal: string;
  rule_score: number;
  anomaly_score: number;
  gnn_score: number;
  taint_score: number;
  taint_path: string[];
  /** JSON strings on this endpoint — the pipeline writes them packed. */
  leads: string;
  contributions: string;
  wallets: number;
  suspicious_merge: boolean;
  /** Queue tiebreakers, in the order fusion/ordering.py applies them after the
   *  composite score. Shown in the queue so the order is legible when several
   *  alerts read the same risk. */
  rule_typologies: number;
  /** Hops from a watchlist seed; 1000000 means no taint path at all. */
  taint_hops: number;
  lead_confidence: number;
  tx_count: number;
  /** The full key as a JSON array, so any consumer orders identically. */
  sort_key: string;
}

export interface AlertsPage {
  alert_threshold: number | null;
  stacker: Record<string, unknown>;
  total: number;
  limit: number;
  offset: number;
  alerts: Alert[];
}

export interface Stats {
  rows: number;
  transactions: number;
  total_entities: number;
  total_alerts: number;
  alerts_by_pattern_type: Record<string, number>;
  avg_confidence: number;
  features: {
    origin_estimation: {
      status: "ok" | "degraded";
      reason: string;
      multi_hop_transactions: number;
      mean_observations_per_transaction: number;
      estimator: string;
    };
  };
  stacker: Record<string, unknown>;
}

export interface EntityDetail {
  entity_id: string;
  entity_type: "cluster" | "wallet";
  wallets: string[];
  flag: string | null;
  features: Record<string, number | string | null>;
  scores: {
    risk_score: number | null;
    rule_score: number | null;
    anomaly_score: number | null;
    gnn_score: number | null;
    taint_score: number | null;
    top_signal: string | null;
    contributions: Record<string, number>;
  };
  alerted: boolean;
  pattern_types: string[];
  reason: string | null;
  evidence: string[];
  taint_path: string[];
  leads: Lead[];
  fingerprints?: FingerprintDistribution & {
    cluster_confidence: number;
    conflicts: { txid: string; heuristic: string; fingerprints: Record<string, string[]> }[];
    note: string;
    console_display?: boolean;
  };
  caveat: string;
}

export type GraphNodeType = "wallet" | "transaction" | "ip";

export interface GraphElements {
  nodes: { data: Record<string, unknown> & { id: string; type?: GraphNodeType } }[];
  edges: { data: Record<string, unknown> & { id: string; source: string; target: string } }[];
}

export interface EntityGraph {
  entity_id: string;
  hops: number;
  truncated: boolean;
  counts: { wallets: number; transactions: number; ips: number };
  layout: { name: string };
  elements: GraphElements;
}

export interface Propagation {
  txid: string;
  estimated_origin: string | null;
  ip_class: IpClass;
  confidence: number;
  attribution_confidence: number;
  estimator: string;
  degraded: boolean;
  low_confidence_origin: boolean;
  anonymized_entry_point: boolean;
  probability: number;
  calibration_basis: string;
  validity: Validity;
  answer: Answer | null;
  n_observations: number;
  runner_ups: { ip: string; score: number }[];
  caveat: string;
  layout: { name: string; roots: string[] };
  elements: GraphElements;
}

/** engines/correlation/profile.py — the reverse direction. A profile describes
 *  a peer's observed network behaviour; it never names who operates it. */
export interface OriginClaim {
  txid: string;
  capture_id: string;
  observer: string;
  probability: number;
  calibration_basis: string;
  tier: Validity["tier"];
  validity: Validity;
  answer: Answer;
  statement: string;
}

export interface EvidenceRow {
  source: string;
  txid: string;
  row?: number;
  timestamp?: string;
  src?: string;
  dst?: string;
  capture_id?: string;
  peer?: string;
  announce_ts?: string;
  capture_source?: string;
}

export interface LinkedCluster {
  cluster: string;
  cluster_id: string;
  basis: "origination" | "correlation lead" | "both";
  confidence: number;
  confidence_rule: string;
  statement: string;
  evidence_chain: string;
  evidence: {
    basis: string;
    txid: string;
    probability: number;
    tier?: string;
    rows: EvidenceRow[];
    inputs: string[];
  }[];
  evidence_total: number;
}

export interface Vantage {
  source: string;
  capture_id: string | null;
  capture_source: string | null;
  provenance: string | null;
  observer: string[];
  direction: string[];
  vantage: string;
  announcements: number;
  first_seen: string | null;
  last_seen: string | null;
}

export interface Timing {
  sufficient: boolean;
  announcements: number;
  threshold: number;
  statement: string;
  announce_rate_per_min?: number | null;
  active_hours_utc?: number[];
  inter_announcement_s?: Record<"mean" | "median" | "stdev" | "min" | "max", number | null>;
}

export interface ClientHistory {
  value: string | number;
  hex?: string;
  first_seen: string | null;
  last_seen: string | null;
  captures: string[];
}

export interface PeerProfile {
  subject: string;
  peer: string;
  kind: "ip" | "onion_identity";
  header: { simulated_only: boolean; provenance: string[]; statement: string };
  originated: {
    claimed: number;
    by_tier: Record<"PASS" | "QUALIFIED" | "ANNOTATE", number>;
    claims: OriginClaim[];
    basis: string;
  };
  propagation_origin: { count: number; statement: string };
  withheld: {
    count: number;
    by_reason: Record<string, number>;
    items: { txid: string; capture_id: string; tier: string; reason: string; statement: string }[];
  };
  relayed: {
    count: number;
    by_source: Record<string, number>;
    sample: { source: string; txid: string; capture_id: string | null; rank: number | null; candidates: number | null }[];
  };
  timing: Timing;
  clients: { user_agents: ClientHistory[]; services: ClientHistory[]; statement: string | null };
  linked_clusters: LinkedCluster[];
  fingerprints?: {
    originated: FingerprintDistribution & { without_structure: number };
    propagation_origin: FingerprintDistribution & { without_structure: number };
    statement: string;
    console_display?: boolean;
  };
  excluded_links: { txid: string; reason: string; statement: string }[];
  vantage: Vantage[];
  /** Absent on an onion identity: it has no IP to enrich. */
  network?: {
    ip: string;
    asn: number | null;
    asn_org: string | null;
    country: string | null;
    ip_class: IpClass | null;
    basis: string;
  };
  caveat: string;
}

export interface AsnMember {
  peer: string;
  subject: string;
  originated: number;
  by_tier: Record<"PASS" | "QUALIFIED" | "ANNOTATE", number>;
  withheld: number;
  relayed: number;
  linked_clusters: string[];
  timing_sufficient: boolean;
  captures: string[];
  asn_org: string | null;
  country: string | null;
}

export interface AsnProfile {
  subject: string;
  asn: number;
  peers: number;
  header: { simulated_only: boolean; statement: string };
  totals: {
    originated: number;
    by_tier: Record<"PASS" | "QUALIFIED" | "ANNOTATE", number>;
    withheld: number;
    relayed: number;
    linked_clusters: number;
  };
  members: AsnMember[];
  caveat: string;
}

/** The origination model's answer for one txid, per capture. */
export interface CaptureOrigination {
  capture_id: string;
  observer: string;
  capture_source: string;
  provenance: string;
  named_peer: string;
  probability: number;
  calibration_basis: string;
  validity: Validity;
  answered: boolean;
  abstention_reason: string | null;
  answer: Answer | null;
  n_candidates: number;
}

export interface TxOrigination {
  txid: string;
  captures: CaptureOrigination[];
  note: string | null;
}

/** features/fingerprint.py — a construction pattern (how, not which software)
 *  pattern, never a party. */
export interface FingerprintAnswer {
  label: string;
  display: string;
  confidence: number | null;
  unknown_reason: string | null;
  ranked: { label: string; display: string; confidence: number; posterior: number }[];
  tells: Record<string, string | null>;
  observed_tells: string[];
  condition?: "full" | "structural";
  basis?: string;
  /** "The label describes how the transaction was built, not which software built it." */
  statement?: string;
  novelty?: { score: number; threshold: number | null; novel: boolean };
}

export interface FingerprintDistribution {
  transactions: number;
  statement?: string;
  labels: { label: string; display: string; count: number; share: number }[];
}

export interface TxFingerprint extends FingerprintAnswer {
  txid: string;
  /** config features.fingerprint.console_display: show by default or not. */
  console_display?: boolean;
}

/** fusion/actors.py — clusters joined to peer identities by evidence. A name
 *  like "actor A-17" is a handle; it never implies a person or organisation. */
export interface ActorLink {
  peer: string;
  peer_kind: "IP" | "onion identity";
  ip_class: string | null;
  cluster_id: string;
  basis: "origination" | "correlation lead" | "both";
  confidence: number;
  origin_tiers: string[];
  evidence_total: number;
  evidence_chain: string;
  evidence: { basis: string; txid: string; probability: number; tier?: string }[];
  /** Whether this link may merge clusters into one actor (docs/ACTORS.md). */
  joins: boolean;
}

export interface Actor {
  actor_id: string;
  name: string;
  risk_score: number;
  membership_confidence: number;
  confidence_basis: string;
  members: string[];
  peers: string[];
  anchor: string;
  member_detail: { entity_id: string; entity_risk: number; weight: number; alerted: boolean }[];
  links: ActorLink[];
  origin_tiers: string[];
  alerted: boolean;
  statement: string;
}

export interface ActorsPage {
  alert_threshold: number | null;
  statement: string | null;
  total: number;
  actors: Actor[];
}

export interface ActorDetail extends Actor {
  drill_down: { entities: Record<string, string>; peers: Record<string, string>; transactions: string[] };
  custody: { seq: number | null };
}
