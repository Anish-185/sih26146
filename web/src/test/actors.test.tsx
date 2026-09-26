/** The queue shows actors by default, toggles back to entities, and names an
 *  actor as a handle, never as a person or organisation. */
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import type { Actor, ActorDetail } from "../api/types";
import { Actor as ActorPage } from "../pages/Actor";
import { Alerts } from "../pages/Alerts";

afterEach(() => vi.restoreAllMocks());

const actor: Actor = {
  actor_id: "A-17", name: "actor A-17", risk_score: 0.91, membership_confidence: 0.38,
  confidence_basis: "the weakest link joining the actor", members: ["c1", "c2"], peers: ["10.0.0.7"],
  anchor: "c1",
  member_detail: [
    { entity_id: "c1", entity_risk: 0.9, weight: 1, alerted: true },
    { entity_id: "c2", entity_risk: 0.2, weight: 0.38, alerted: false },
  ],
  links: [{ peer: "10.0.0.7", peer_kind: "IP", ip_class: "residential_or_unknown", cluster_id: "c2",
            basis: "both", confidence: 0.38, origin_tiers: ["PASS"], evidence_total: 1,
            evidence_chain: "raw rows -> transaction -> input addresses -> cluster",
            evidence: [{ basis: "origination", txid: "e".repeat(64), probability: 0.7, tier: "PASS" }],
            joins: true }],
  origin_tiers: ["PASS"], alerted: true,
  statement: "an actor is a group of addresses and network identities linked by evidence; it does not name or imply a person or an organisation",
};

function queue(path = "/alerts") {
  vi.spyOn(api, "actors").mockResolvedValue({ alert_threshold: 0.5, statement: actor.statement,
                                              total: 1, actors: [actor] });
  vi.spyOn(api, "alerts").mockResolvedValue({ alert_threshold: 0.5, stacker: {}, total: 0, limit: 200,
                                              offset: 0, filters: {}, alerts: [] } as never);
  render(<MemoryRouter initialEntries={[path]}><Routes>
    <Route path="/alerts" element={<Alerts />} /></Routes></MemoryRouter>);
}

describe("actor queue", () => {
  it("shows actors by default, with the statement and linked peers", async () => {
    queue();
    expect(await screen.findByRole("link", { name: "actor A-17" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Actor queue" })).toBeInTheDocument();
    expect(screen.getByText(/does not name or imply a person or an organisation/)).toBeInTheDocument();
    expect(screen.getByText(/10.0.0.7/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/owner|operated by|belongs to|suspect/i);
  });

  it("toggles back to the entity queue", async () => {
    queue();
    await screen.findByRole("link", { name: "actor A-17" });
    fireEvent.click(screen.getByRole("button", { name: /entities/ }));
    expect(await screen.findByRole("heading", { name: "Alert queue" })).toBeInTheDocument();
  });
});

describe("actor page", () => {
  it("drills down to entities, peers and transactions", async () => {
    const detail: ActorDetail = { ...actor, custody: { seq: 4 },
      drill_down: { entities: { c1: "/entities/c1" }, peers: {}, transactions: [] } };
    vi.spyOn(api, "actor").mockResolvedValue(detail);
    render(<MemoryRouter initialEntries={["/actors/A-17"]}><Routes>
      <Route path="/actors/:id" element={<ActorPage />} /></Routes></MemoryRouter>);
    expect(await screen.findByRole("heading", { name: "actor A-17" })).toBeInTheDocument();
    expect(screen.getByText(/by both, confidence 0.38/)).toBeInTheDocument();
    expect(document.querySelector('a[href^="/tx/"]')).not.toBeNull();
    expect(document.querySelector('a[href="/entities/c1"]')).not.toBeNull();
  });
});
