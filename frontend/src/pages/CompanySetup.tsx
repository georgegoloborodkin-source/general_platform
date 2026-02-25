/**
 * Company Setup — the core onboarding step
 *
 * The MD/boss pastes a description of their company and the backend
 * auto-generates a custom system prompt for the AI. This prompt is
 * stored on the organization and shared with all team members.
 */
import { useState } from "react";
import { setupCompany } from "../utils/api";

const EXAMPLE_DESCRIPTIONS = [
  {
    label: "Trading Firm",
    text: "We are a quantitative trading firm focused on commodities and fixed income derivatives. Our team of 15 manages $500M AUM across systematic and discretionary strategies. Key metrics we track: PnL, Sharpe ratio, max drawdown, VaR. We deal with strategy documents, risk reports, market research, and compliance filings.",
  },
  {
    label: "Marketing Agency",
    text: "We are a digital marketing agency with 30 people serving mid-market B2B SaaS clients. We manage campaigns across Google Ads, LinkedIn, content marketing, and ABM. Key metrics: CAC, ROAS, pipeline generated, MQL-to-SQL conversion rate. We work with campaign briefs, analytics reports, creative decks, and client proposals.",
  },
  {
    label: "Consulting Firm",
    text: "We are a management consulting firm specializing in digital transformation for financial services. Our team of 50 consultants works on 6-month engagements. We track: utilization rate, project margin, client NPS, revenue per consultant. Documents include project proposals, client deliverables, market analyses, and case studies.",
  },
  {
    label: "Real Estate",
    text: "We are a commercial real estate investment firm managing a $200M portfolio of office and retail properties across the Southeast US. We track: cap rate, NOI, occupancy, rent per sqft, IRR. Our documents include property analyses, investment memos, lease agreements, market reports, and appraisals.",
  },
];

interface Props {
  organizationId: string;
  companyName: string;
  invitationCode?: string | null;
  onComplete: () => void;
}

export default function CompanySetup({ organizationId, companyName, invitationCode, onComplete }: Props) {
  const [description, setDescription] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedPrompt, setGeneratedPrompt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [codeCopied, setCodeCopied] = useState(false);

  const handleGenerate = async () => {
    if (!description.trim()) return;
    setIsGenerating(true);
    setError(null);

    try {
      const result = await setupCompany({
        organizationId,
        companyDescription: description.trim(),
        companyName,
      });
      setGeneratedPrompt(result.system_prompt);
    } catch (err: any) {
      setError(err.message || "Failed to generate AI context");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleContinue = () => {
    onComplete();
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-slate-950">
      <div className="w-full max-w-3xl border border-slate-700/60 bg-slate-900/50 backdrop-blur-sm rounded-2xl shadow-xl shadow-black/20 overflow-hidden">
        {/* Header */}
        <div className="text-center border-b border-slate-700/60 bg-slate-800/30 px-6 py-6">
          <h1 className="text-2xl font-bold text-white">
            {generatedPrompt ? "AI Context Generated!" : "Tell us about your company"}
          </h1>
          <p className="text-slate-400 text-base mt-1">
            {generatedPrompt
              ? "Your AI assistant is now customized for your team"
              : "Paste a description and we'll customize the AI for your team"
            }
          </p>
        </div>

        <div className="p-6 space-y-5">
          {!generatedPrompt ? (
            <>
              {invitationCode && (
                <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5">
                  <div className="flex items-center gap-2 mb-2">
                    <svg className="h-5 w-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <span className="text-emerald-200 font-semibold text-sm">Your team invitation code</span>
                  </div>
                  <p className="text-slate-400 text-xs mb-3">
                    Share this code with your team members so they can join your organization.
                  </p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 rounded-lg border border-emerald-500/30 bg-slate-800/60 px-4 py-3 text-center font-mono text-xl tracking-widest text-white select-all">
                      {invitationCode}
                    </div>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(invitationCode);
                        setCodeCopied(true);
                        setTimeout(() => setCodeCopied(false), 2000);
                      }}
                      className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 px-4 py-3 text-sm font-medium transition-all"
                    >
                      {codeCopied ? "Copied!" : "Copy"}
                    </button>
                  </div>
                </div>
              )}

              {/* Examples */}
              <div>
                <p className="text-xs text-slate-500 mb-2 font-medium uppercase tracking-wider">
                  Quick examples (click to use)
                </p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLE_DESCRIPTIONS.map((ex) => (
                    <button
                      key={ex.label}
                      onClick={() => setDescription(ex.text)}
                      className="px-3 py-1.5 text-xs rounded-lg border border-slate-600 bg-slate-800/40 text-slate-300 hover:border-amber-500/30 hover:bg-slate-700/50 hover:text-white transition-all"
                    >
                      {ex.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Textarea */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Company Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe your company: what you do, your team size, what documents you work with, key metrics you track, types of relationships/connections that matter..."
                  className="w-full min-h-[200px] rounded-xl border border-slate-600 bg-slate-800/50 text-white placeholder:text-slate-500 p-4 text-sm resize-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30 outline-none"
                />
                <p className="text-xs text-slate-500 mt-1">
                  The more detail you provide, the better the AI will understand your context.
                  Include: industry, team size, document types, key metrics, terminology.
                </p>
              </div>

              {error && (
                <div className="p-3 rounded-lg border border-red-500/20 bg-red-500/5 text-red-300 text-sm">
                  {error}
                </div>
              )}

              {/* Generate button */}
              <button
                onClick={handleGenerate}
                disabled={!description.trim() || isGenerating}
                className="w-full rounded-lg bg-amber-500 text-slate-950 hover:bg-amber-400 font-semibold h-12 shadow-[0_2px_12px_-2px_rgba(245,158,11,0.4)] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                {isGenerating ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Generating AI context...
                  </span>
                ) : (
                  "Generate AI Context"
                )}
              </button>
            </>
          ) : (
            <>
              {/* Success: show the generated prompt */}
              <div className="p-4 rounded-xl border border-amber-500/20 bg-amber-500/5">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="h-5 w-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-amber-200 font-semibold text-sm">AI context generated and saved</span>
                </div>
                <p className="text-slate-400 text-xs">
                  All team members who join your organization will automatically use this context.
                </p>
              </div>

              {invitationCode && (
                <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5">
                  <div className="flex items-center gap-2 mb-2">
                    <svg className="h-5 w-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <span className="text-emerald-200 font-semibold text-sm">Invite your team</span>
                  </div>
                  <p className="text-slate-400 text-xs mb-3">
                    Share this code with your team members. They sign up, choose <strong className="text-white">Team Member</strong>, and enter this code to join your organization.
                  </p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 rounded-lg border border-emerald-500/30 bg-slate-800/60 px-4 py-3 text-center font-mono text-xl tracking-widest text-white select-all">
                      {invitationCode}
                    </div>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(invitationCode);
                        setCodeCopied(true);
                        setTimeout(() => setCodeCopied(false), 2000);
                      }}
                      className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 px-4 py-3 text-sm font-medium transition-all"
                    >
                      {codeCopied ? "Copied!" : "Copy"}
                    </button>
                  </div>
                </div>
              )}

              <div>
                <label className="block text-xs text-slate-500 mb-1 font-medium uppercase tracking-wider">
                  Generated AI System Prompt (preview)
                </label>
                <div className="max-h-[300px] overflow-y-auto rounded-xl border border-slate-700 bg-slate-800/30 p-4 text-sm text-slate-300 font-mono whitespace-pre-wrap">
                  {generatedPrompt}
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => { setGeneratedPrompt(null); }}
                  className="flex-1 rounded-lg border border-slate-600 bg-slate-800/50 text-slate-200 hover:bg-slate-700/60 font-medium h-11 transition-all"
                >
                  Edit & Regenerate
                </button>
                <button
                  onClick={handleContinue}
                  className="flex-1 rounded-lg bg-amber-500 text-slate-950 hover:bg-amber-400 font-semibold h-11 shadow-[0_2px_12px_-2px_rgba(245,158,11,0.4)] transition-all"
                >
                  Continue to Dashboard
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
