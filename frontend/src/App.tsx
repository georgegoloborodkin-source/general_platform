import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/hooks/useAuth";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";
import AuthCallback from "./pages/AuthCallback";

const queryClient = new QueryClient();
const PRODUCTION_HOSTNAME = "general-platform.vercel.app";

function useProductionDomainRedirect() {
  if (typeof window === "undefined") return false;
  const { hostname, pathname, search, hash, protocol } = window.location;
  const isVercelPreview = hostname.endsWith(".vercel.app") && hostname !== PRODUCTION_HOSTNAME;
  if (!isVercelPreview) return false;

  // Force previews onto production so auth + API behavior is consistent.
  const target = `${protocol}//${PRODUCTION_HOSTNAME}${pathname}${search}${hash}`;
  window.location.replace(target);
  return true;
}

// If Supabase OAuth redirected to root (or another path) with tokens in the hash, send to callback so session is established
function useOAuthHashRedirect() {
  const hash = typeof window !== "undefined" ? window.location.hash : "";
  const pathname = typeof window !== "undefined" ? window.location.pathname : "";
  if (hash && hash.includes("access_token") && pathname !== "/auth/callback") {
    window.location.replace("/auth/callback" + hash);
    return true;
  }
  return false;
}

const App = () => {
  const redirectingToProduction = useProductionDomainRedirect();
  if (redirectingToProduction) return null;

  const redirecting = useOAuthHashRedirect();
  if (redirecting) return null;

  return (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <div className="app-shell cis-grid-bg cis-mesh-bg">
            <Routes>
              <Route
                path="/"
                element={
                  <ProtectedRoute requireAuth>
                    <Dashboard />
                  </ProtectedRoute>
                }
              />
              <Route path="/login" element={<Login />} />
              <Route path="/auth/callback" element={<AuthCallback />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </BrowserRouter>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
  );
};

export default App;
