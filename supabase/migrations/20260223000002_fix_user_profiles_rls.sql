-- Fix infinite recursion in user_profiles RLS
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run
--
-- The policy "Org members can view each other" reads from user_profiles to decide access, causing recursion.
-- Drop it and rely on "Users can view own profile" for the common case (useAuth fetches own profile).
-- Add INSERT/UPDATE so the app can create/update the current user's profile on sign-in.

-- Drop the recursive policy
DROP POLICY IF EXISTS "Org members can view each other" ON user_profiles;

-- Allow users to insert their own profile (for first-time sign-in / AuthCallback upsert)
CREATE POLICY "Users can insert own profile" ON user_profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

-- Allow users to update their own profile
CREATE POLICY "Users can update own profile" ON user_profiles
  FOR UPDATE USING (auth.uid() = id) WITH CHECK (auth.uid() = id);
