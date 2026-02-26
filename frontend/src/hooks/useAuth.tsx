import { createContext, useContext, useEffect, useRef, useState } from "react";
import { User } from "@supabase/supabase-js";
import { supabase } from "@/integrations/supabase/client";
import { UserProfile } from "@/types";
import { clearGoogleProviderTokens } from "@/utils/googleAuthStorage";

const PROFILE_REFRESH_THROTTLE_MS = 8000;

interface AuthContextType {
  user: User | null;
  profile: UserProfile | null;
  loading: boolean;
  signOut: () => Promise<void>;
  refreshProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const SESSION_RECOVERY_DELAY_MS = 65_000; // After 429, Supabase rate limit often resets ~60s

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const lastProfileRefreshRef = useRef<number>(0);
  const sessionRecoveryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const explicitSignOutRef = useRef(false);

  const refreshProfile = async () => {
    if (!user) {
      setProfile(null);
      return;
    }
    const now = Date.now();
    if (now - lastProfileRefreshRef.current < PROFILE_REFRESH_THROTTLE_MS) {
      return;
    }
    lastProfileRefreshRef.current = now;

    try {
      // Use maybeSingle() to avoid 406 when no row exists (PostgREST returns 406 for .single() with 0 rows)
      const { data, error } = await supabase
        .from('user_profiles')
        .select('*')
        .eq('id', user.id)
        .maybeSingle();

      if (error) throw error;

      if (!data) {
        // No profile yet — create one (first-time sign-in or race after AuthCallback)
        const { error: upsertError } = await supabase
          .from("user_profiles")
          .upsert(
            {
              id: user.id,
              email: user.email,
              full_name: (user.user_metadata as any)?.full_name || (user.user_metadata as any)?.name || "",
              role: "team_member",
            },
            { onConflict: "id", ignoreDuplicates: true }
          );

        if (upsertError) throw upsertError;

        const { data: created, error: createdError } = await supabase
          .from("user_profiles")
          .select("*")
          .eq("id", user.id)
          .maybeSingle();

        if (createdError) throw createdError;
        setProfile(created);
        return;
      }

      setProfile(data);
    } catch (error) {
      console.error("Error fetching profile:", error);
      setProfile(null);
    }
  };

  useEffect(() => {
    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      setLoading(false);
    });

    // Listen for auth changes (avoid refreshing profile on TOKEN_REFRESHED to reduce 429s)
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (session?.user) {
        explicitSignOutRef.current = false;
        if (sessionRecoveryTimeoutRef.current) {
          clearTimeout(sessionRecoveryTimeoutRef.current);
          sessionRecoveryTimeoutRef.current = null;
        }
        setUser(session.user);
        if (event === "TOKEN_REFRESHED") return;
        refreshProfile();
      } else {
        setProfile(null);
        // Session lost (could be 429 rate limit). If user didn't explicitly sign out, try recovery after delay so they aren't kicked out.
        if (explicitSignOutRef.current) {
          setUser(null);
          return;
        }
        if (sessionRecoveryTimeoutRef.current) return; // Already scheduled
        sessionRecoveryTimeoutRef.current = setTimeout(async () => {
          sessionRecoveryTimeoutRef.current = null;
          if (explicitSignOutRef.current) return;
          try {
            const { data } = await supabase.auth.refreshSession();
            if (data?.session) {
              setUser(data.session.user);
              refreshProfile();
            } else {
              setUser(null);
            }
          } catch {
            setUser(null);
          }
        }, SESSION_RECOVERY_DELAY_MS);
        // Do not set user to null here — keep them on the page until recovery attempt completes
      }
    });

    return () => {
      subscription.unsubscribe();
      if (sessionRecoveryTimeoutRef.current) {
        clearTimeout(sessionRecoveryTimeoutRef.current);
        sessionRecoveryTimeoutRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (user) {
      refreshProfile();
    }
  }, [user]);

  const signOut = async () => {
    explicitSignOutRef.current = true;
    if (sessionRecoveryTimeoutRef.current) {
      clearTimeout(sessionRecoveryTimeoutRef.current);
      sessionRecoveryTimeoutRef.current = null;
    }
    clearGoogleProviderTokens();
    await supabase.auth.signOut();
    setUser(null);
    setProfile(null);
  };

  return (
    <AuthContext.Provider value={{ user, profile, loading, signOut, refreshProfile }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

