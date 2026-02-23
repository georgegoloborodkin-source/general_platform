import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useState } from "react";
import RoleSelection from "./pages/RoleSelection";
import CompanySetup from "./pages/CompanySetup";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const [orgId, setOrgId] = useState<string | null>(null);

  if (!isLoggedIn) {
    return (
      <Login onLogin={(uid) => {
        setUserId(uid);
        setIsLoggedIn(true);
      }} />
    );
  }

  if (!orgId) {
    return (
      <RoleSelection
        userId={userId!}
        onComplete={(oid) => setOrgId(oid)}
      />
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard orgId={orgId} userId={userId!} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
