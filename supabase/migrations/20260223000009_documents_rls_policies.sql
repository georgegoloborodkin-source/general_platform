-- RLS policies for documents table (required for Drive sync and app access)
-- Without these, INSERT/SELECT/UPDATE/DELETE are denied (42501).
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

-- Ensure user_event_ids() exists (from 20260223000007)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON p.pronamespace = n.oid WHERE n.nspname = 'public' AND p.proname = 'user_event_ids') THEN
    CREATE OR REPLACE FUNCTION public.user_event_ids()
    RETURNS SETOF uuid
    LANGUAGE sql
    STABLE
    SECURITY DEFINER
    SET search_path = public
    AS $fn$
      SELECT e.id FROM events e
      WHERE e.organization_id IN (SELECT organization_id FROM user_profiles WHERE id = auth.uid())
    $fn$;
  END IF;
END $$;

DROP POLICY IF EXISTS "Members can manage documents" ON public.documents;
CREATE POLICY "Members can manage documents" ON public.documents
  FOR ALL
  USING (event_id IN (SELECT user_event_ids()))
  WITH CHECK (event_id IN (SELECT user_event_ids()));
