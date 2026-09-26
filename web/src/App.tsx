import { Suspense, lazy } from "react";
import { Route, Routes } from "react-router-dom";
import { ToastHost } from "./components/Toasts";
import { Home } from "./pages/Home";
import { Alerts } from "./pages/Alerts";
import { Entity } from "./pages/Entity";
import { Transaction } from "./pages/Transaction";
import { RedTeam } from "./pages/RedTeam";
import { Monitor } from "./pages/Monitor";
import { Actor } from "./pages/Actor";
import { Custody } from "./pages/Custody";
import { Asn, Peer, Peers } from "./pages/Peer";

// The investigation graph pulls in Cytoscape and its extensions — the heaviest
// thing the console loads, and only this route needs it.
const Investigate = lazy(() =>
  import("./pages/Investigate").then((m) => ({ default: m.Investigate })),
);

export function App() {
  return (
    <ToastHost>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/entities/:id" element={<Entity />} />
        <Route path="/actors/:id" element={<Actor />} />
        <Route path="/tx/:txid" element={<Transaction />} />
        <Route path="/peers" element={<Peers />} />
        <Route path="/peers/:peer" element={<Peer />} />
        <Route path="/asns/:asn" element={<Asn />} />
        <Route path="/redteam" element={<RedTeam />} />
        <Route path="/monitor" element={<Monitor />} />
        <Route path="/custody" element={<Custody />} />
        <Route
          path="/investigate"
          element={
            <Suspense fallback={<div className="shell" aria-busy="true" />}>
              <Investigate />
            </Suspense>
          }
        />
        <Route path="*" element={<Home />} />
      </Routes>
    </ToastHost>
  );
}
