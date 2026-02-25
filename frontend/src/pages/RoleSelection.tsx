/**
 * Role Selection → Company Setup (Admin) or Code Entry (Team Member)
 *
 * Flow:
 *   Admin: Select role → Create org → Paste company description → Dashboard
 *   Team:  Select role → Enter invitation code → Dashboard (inherits org context)
 */
import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import CompanySetup from "./CompanySetup";

interface Props {
  userId: string;
  onComplete: (orgId: string) => void;
}

type Step = "role" | "admin_name" | "admin_setup" | "team_code" | "done";

export default function RoleSelection({ userId, onComplete }: Props) {
  const [step, setStep] = useState<Step>("role");
  const [isSaving, setIsSaving] = useState(false);
  const [orgName, setOrgName] = useState("");
  const [orgId, setOrgId] = useState<string | null>(null);
  const [invitationCode, setInvitationCode] = useState("");
  const [createdCode, setCreatedCode] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const cardClass = "w-full max-w-2xl border border-slate-700/60 bg-slate-900/50 backdrop-blur-sm rounded-2xl shadow-xl shadow-black/20 overflow-hidden";
  const btnPrimary = "w-full rounded-lg bg-amber-500 text-slate-950 hover:bg-amber-400 font-semibold h-11 shadow-[0_2px_12px_-2px_rgba(245,158,11,0.4)] disabled:opacity-50 transition-all";
  const btnSecondary = "w-full rounded-lg border-slate-600 bg-slate-800/50 text-slate-200 hover:bg-slate-700/60 font-medium h-11 transition-all";

  const handleCreateOrg = async () => {
    if (!orgName.trim()) return;
    setIsSaving(true);
    setError(null);

    try {
      const slug = orgName.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      const { data, error: rpcError } = await supabase.rpc("ensure_user_organization", {
        org_name: orgName.trim(),
        org_slug: slug || "org",
      });
      if (rpcError) throw new Error(rpcError.message);
      if (!data?.id) throw new Error("Organization creation failed — no ID returned.");

      // Update role to managing_partner for the admin
      await supabase.from("user_profiles").update({ role: "managing_partner" }).eq("id", userId);

      setOrgId(data.id);
      setCreatedCode(data.invitation_code || null);
      setStep("admin_setup");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  const handleJoinOrg = async () => {
    if (!invitationCode.trim()) return;
    setIsSaving(true);
    setError(null);

    try {
      const { data, error: rpcError } = await supabase.rpc("join_organization_by_code", {
        code: invitationCode.trim(),
      });
      if (rpcError) throw new Error(rpcError.message);
      if (!data?.id) throw new Error("Could not join organization — check your code and try again.");

      setOrgId(data.id);
      onComplete(data.id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsSaving(false);
    }
  };

  // Admin: company setup is done
  if (step === "admin_setup" && orgId) {
    return (
      <CompanySetup
        organizationId={orgId}
        companyName={orgName}
        onComplete={() => onComplete(orgId)}
      />
    );
  }

  // Admin: enter company name
  if (step === "admin_name") {
    return (
      <div className="flex items-center justify-center min-h-screen p-4 bg-slate-950">
        <div className={cardClass}>
          <div className="text-center border-b border-slate-700/60 bg-slate-800/30 px-6 py-6">
            <h1 className="text-2xl font-bold text-white">Create Your Organization</h1>
            <p className="text-slate-400 text-base mt-1">
              Enter your company name to get started
            </p>
          </div>
          <div className="p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Company Name</label>
              <input
                value={orgName}
                onChange={(e) => setOrgName(e.target.value)}
                placeholder="e.g., Apex Trading, Summit Consulting"
                className="w-full rounded-lg border border-slate-600 bg-slate-800/50 text-white placeholder:text-slate-500 px-4 py-3 text-sm focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30 outline-none"
                onKeyDown={(e) => { if (e.key === "Enter" && orgName.trim()) handleCreateOrg(); }}
              />
            </div>
            {error && (
              <div className="p-3 rounded-lg border border-red-500/20 bg-red-500/5 text-red-300 text-sm">{error}</div>
            )}
            <button onClick={handleCreateOrg} disabled={!orgName.trim() || isSaving} className={btnPrimary}>
              {isSaving ? "Creating..." : "Create Organization"}
            </button>
            <button onClick={() => setStep("role")} className={btnSecondary}>Back</button>
          </div>
        </div>
      </div>
    );
  }

  // Team: enter invitation code
  if (step === "team_code") {
    return (
      <div className="flex items-center justify-center min-h-screen p-4 bg-slate-950">
        <div className={cardClass}>
          <div className="text-center border-b border-slate-700/60 bg-slate-800/30 px-6 py-6">
            <h1 className="text-2xl font-bold text-white">Join Your Team</h1>
            <p className="text-slate-400 text-base mt-1">
              Enter the invitation code from your admin
            </p>
          </div>
          <div className="p-6 space-y-4">
            <div className="p-3 rounded-lg border border-amber-500/20 bg-amber-500/5 text-slate-300 text-sm">
              <span className="text-amber-400 font-semibold">Note:</span> Your admin should give you an invitation code (e.g., ORB-ABC123). Once you join, you'll automatically share the same AI context as your team.
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Invitation Code</label>
              <input
                value={invitationCode}
                onChange={(e) => setInvitationCode(e.target.value.toUpperCase())}
                placeholder="ORB-ABC123"
                className="w-full rounded-lg border border-slate-600 bg-slate-800/50 text-white placeholder:text-slate-500 px-4 py-3 text-center font-mono text-lg tracking-wider focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30 outline-none"
                onKeyDown={(e) => { if (e.key === "Enter" && invitationCode.trim()) handleJoinOrg(); }}
              />
            </div>
            {error && (
              <div className="p-3 rounded-lg border border-red-500/20 bg-red-500/5 text-red-300 text-sm">{error}</div>
            )}
            <button onClick={handleJoinOrg} disabled={!invitationCode.trim() || isSaving} className={btnPrimary}>
              {isSaving ? "Joining..." : "Join Organization"}
            </button>
            <button onClick={() => setStep("role")} className={btnSecondary}>Back</button>
          </div>
        </div>
      </div>
    );
  }

  // Role selection
  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-slate-950">
      <div className={cardClass}>
        <div className="text-center border-b border-slate-700/60 bg-slate-800/30 px-6 py-6">
          <h1 className="text-2xl font-bold text-white">Welcome to Orbit Platform</h1>
          <p className="text-slate-400 text-base mt-1">
            Choose your role to get started
          </p>
        </div>
        <div className="p-6 space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <button
              onClick={() => setStep("admin_name")}
              className="h-auto p-6 flex flex-col items-center justify-center space-y-3 rounded-xl border border-slate-600 bg-slate-800/40 text-slate-200 hover:bg-slate-700/50 hover:border-amber-500/30 hover:text-white transition-all duration-200"
            >
              <svg className="h-12 w-12 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
              <div className="text-center">
                <div className="font-semibold text-lg text-white">Admin / Boss</div>
                <div className="text-sm text-slate-400 mt-1">
                  Create your organization, describe your company, and invite your team
                </div>
              </div>
            </button>

            <button
              onClick={() => setStep("team_code")}
              className="h-auto p-6 flex flex-col items-center justify-center space-y-3 rounded-xl border border-slate-600 bg-slate-800/40 text-slate-200 hover:bg-slate-700/50 hover:border-amber-500/30 hover:text-white transition-all duration-200"
            >
              <svg className="h-12 w-12 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              <div className="text-center">
                <div className="font-semibold text-lg text-white">Team Member</div>
                <div className="text-sm text-slate-400 mt-1">
                  Enter invitation code from your admin to join the team
                </div>
              </div>
            </button>
          </div>

          <div className="border-t border-slate-700/60 pt-4">
            <div className="p-3 rounded-lg border border-slate-600 bg-slate-800/30 text-slate-300 text-sm">
              <span className="text-amber-400 font-semibold">How it works:</span>
              <br />
              <strong className="text-white">Admins</strong> create the organization, paste a description of what the company does, and the AI automatically adapts to your industry.
              <br />
              <strong className="text-white">Team Members</strong> enter the code to join — they'll share the same AI context.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
