/**
 * Onboarding page — shown after Google login when user has no organization.
 * Wraps RoleSelection and navigates to Dashboard on completion.
 */
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import RoleSelection from "./RoleSelection";

export default function Onboarding() {
  const navigate = useNavigate();
  const { user, refreshProfile } = useAuth();

  const handleComplete = async (_orgId: string) => {
    await refreshProfile();
    navigate("/", { replace: true });
  };

  if (!user) return null;

  return <RoleSelection userId={user.id} onComplete={handleComplete} />;
}
