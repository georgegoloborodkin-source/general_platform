import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LoginButton } from "@/components/Auth/LoginButton";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Brain } from "lucide-react";

export default function Login() {
  const navigate = useNavigate();

  useEffect(() => {
    const checkSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        navigate("/");
      }
    };
    checkSession();
  }, [navigate]);

  return (
    <div className="min-h-screen bg-white flex flex-col cis-app">
      <div className="fixed inset-0 cis-grid-bg cis-mesh-bg pointer-events-none" />
      <div className="flex-1 flex items-center justify-center px-4 py-12 relative z-10">
        <div className="w-full max-w-md">
          <Card className="border border-slate-200 bg-white shadow-lg rounded-2xl overflow-hidden">
            <CardHeader className="border-b border-slate-200 px-8 pt-8 pb-6">
              <div className="flex items-center gap-4">
                <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-blue-600/15 text-blue-600">
                  <Brain className="h-7 w-7" />
                </div>
                <div>
                  <CardTitle className="text-2xl font-bold text-slate-900">Welcome</CardTitle>
                  <CardDescription className="text-slate-500 text-sm mt-1">
                    Sign in to access your team platform.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-8 pb-8 px-8">
              <LoginButton />
              <p className="mt-6 text-center text-xs text-slate-500">
                By signing in, you agree to our Terms of Service and Privacy Policy.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
