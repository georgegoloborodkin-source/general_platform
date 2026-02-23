import { useState } from "react";

interface Props {
  onLogin: (userId: string) => void;
}

export default function Login({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async () => {
    if (!email.trim()) return;
    setIsLoading(true);
    // TODO: Integrate with Supabase Auth (Google OAuth, magic link, etc.)
    // For now, simulate login
    setTimeout(() => {
      onLogin(crypto.randomUUID());
      setIsLoading(false);
    }, 500);
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4 bg-slate-950">
      <div className="w-full max-w-md border border-slate-700/60 bg-slate-900/50 backdrop-blur-sm rounded-2xl shadow-xl shadow-black/20 overflow-hidden">
        <div className="text-center border-b border-slate-700/60 bg-slate-800/30 px-6 py-8">
          <div className="text-4xl font-black text-white tracking-tight mb-1">Orbit</div>
          <div className="text-amber-400 font-medium text-sm tracking-wider uppercase">Platform</div>
          <p className="text-slate-400 text-sm mt-3">
            Intelligence system for any company
          </p>
        </div>
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLogin()}
              placeholder="you@company.com"
              className="w-full rounded-lg border border-slate-600 bg-slate-800/50 text-white placeholder:text-slate-500 px-4 py-3 text-sm focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/30 outline-none"
            />
          </div>
          <button
            onClick={handleLogin}
            disabled={!email.trim() || isLoading}
            className="w-full rounded-lg bg-amber-500 text-slate-950 hover:bg-amber-400 font-semibold h-11 shadow-[0_2px_12px_-2px_rgba(245,158,11,0.4)] disabled:opacity-50 transition-all"
          >
            {isLoading ? "Signing in..." : "Continue"}
          </button>
          <p className="text-center text-xs text-slate-500">
            We'll connect Supabase Auth (Google, GitHub, etc.) here
          </p>
        </div>
      </div>
    </div>
  );
}
