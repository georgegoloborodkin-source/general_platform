import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft } from "lucide-react";

export default function AuthCallback() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const handleAuthCallback = async () => {
      try {
        // Supabase redirects with tokens in the URL hash. Parse and set session so the user is stored.
        const hash = window.location.hash;
        if (hash) {
          const params = new URLSearchParams(hash.replace(/^#/, ""));
          const access_token = params.get("access_token");
          const refresh_token = params.get("refresh_token");
          if (access_token) {
            const { data, error } = await supabase.auth.setSession({
              access_token,
              refresh_token: refresh_token || undefined,
            });
            if (error) throw error;
            if (data?.session) {
              window.history.replaceState(null, "", window.location.pathname);
              const user = data.session.user;
              const { error: profileError } = await supabase.from("user_profiles").select("id").eq("id", user.id).single();
              if (profileError?.code === "PGRST116") {
                await supabase.from("user_profiles").upsert({
                  id: user.id,
                  email: user.email,
                  full_name: user.user_metadata?.full_name || user.user_metadata?.name || "",
                  role: "team_member",
                }, { onConflict: "id", ignoreDuplicates: true });
              }
              toast({ title: "Successfully signed in!", description: "Welcome to the platform." });
              await new Promise((r) => setTimeout(r, 200));
              navigate("/");
              return;
            }
          }
        }

        let session = null;
        let attempts = 0;
        const maxAttempts = 10;

        while (!session && attempts < maxAttempts) {
          const { data, error } = await supabase.auth.getSession();
          if (error) throw error;
          if (data?.session) {
            session = data.session;
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 300 * (attempts + 1)));
          attempts++;
        }

        if (!session) {
          const urlParams = new URLSearchParams(window.location.search);
          const errorParam = urlParams.get("error");
          const errorDescription = urlParams.get("error_description");
          if (errorParam) throw new Error(errorDescription || errorParam);
          throw new Error("Session not established. Please try signing in again.");
        }

        const user = session.user;

        const { data: profile, error: profileError } = await supabase
          .from("user_profiles")
          .select("*")
          .eq("id", user.id)
          .single();

        if (profileError && profileError.code === "PGRST116") {
          const { error: upsertError } = await supabase.from("user_profiles").upsert(
            {
              id: user.id,
              email: user.email,
              full_name: user.user_metadata?.full_name || user.user_metadata?.name || "",
              role: "team_member",
            },
            { onConflict: "id", ignoreDuplicates: true }
          );
          if (upsertError) throw upsertError;
        }

        window.history.replaceState(null, "", window.location.pathname);

        toast({
          title: "Successfully signed in!",
          description: "Welcome to the platform.",
        });

        await new Promise((resolve) => setTimeout(resolve, 200));
        navigate("/");
      } catch (error: any) {
        console.error("Auth callback error:", error);
        toast({
          title: "Authentication error",
          description: error.message || "Failed to complete sign in",
          variant: "destructive",
        });
        setErrorMessage(error.message || "Failed to complete sign in.");
      }
    };

    handleAuthCallback();
  }, [navigate, toast]);

  return (
    <div className="flex min-h-screen items-center justify-center px-4 cis-app">
      <div className="text-center">
        {errorMessage ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-8 max-w-md space-y-4 shadow-lg">
            <div className="text-lg font-semibold text-slate-900">Sign-in failed</div>
            <div className="text-sm text-slate-500">{errorMessage}</div>
            <Button
              onClick={() => navigate("/login")}
              className="rounded-lg bg-blue-600 text-white hover:bg-blue-500 font-semibold"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Login
            </Button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="h-14 w-14 rounded-2xl bg-blue-600/15 flex items-center justify-center">
              <Loader2 className="h-7 w-7 animate-spin text-blue-600" />
            </div>
            <p className="text-slate-500 font-medium">Completing sign in...</p>
          </div>
        )}
      </div>
    </div>
  );
}
