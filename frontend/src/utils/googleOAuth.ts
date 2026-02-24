import { supabase } from "@/integrations/supabase/client";

const PRODUCTION_ORIGIN = "https://general-platform.vercel.app";

export function getGoogleOAuthRedirectTo(): string {
  if (typeof window !== "undefined" && window.location?.hostname === "localhost") {
    return `${window.location.origin}/auth/callback`;
  }
  return `${PRODUCTION_ORIGIN}/auth/callback`;
}

/**
 * Trigger Google OAuth with Drive + Gmail scopes. Use when user needs to (re-)grant Google Drive access.
 * Redirects the page to Google; after consent, user returns to /auth/callback.
 */
export async function triggerGoogleOAuthForDrive(): Promise<void> {
  const redirectTo = getGoogleOAuthRedirectTo();
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: {
      redirectTo,
      scopes: "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/gmail.readonly",
      queryParams: {
        access_type: "offline",
        prompt: "consent",
      },
    },
  });
  if (error) throw error;
  if (data?.url) {
    window.location.href = data.url;
  }
}
