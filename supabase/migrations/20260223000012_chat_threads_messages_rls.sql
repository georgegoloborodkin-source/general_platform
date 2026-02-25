-- RLS policies for chat_threads and chat_messages (required for chat to save and load)
-- Without these, INSERT/SELECT are denied and nothing appears in the tables.
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

-- chat_threads: org members can view and insert threads for their events
DROP POLICY IF EXISTS "Members can view org chat threads" ON public.chat_threads;
DROP POLICY IF EXISTS "Members can insert org chat threads" ON public.chat_threads;
CREATE POLICY "Members can view org chat threads" ON public.chat_threads
  FOR SELECT USING (event_id IN (SELECT user_event_ids()));
CREATE POLICY "Members can insert org chat threads" ON public.chat_threads
  FOR INSERT WITH CHECK (event_id IN (SELECT user_event_ids()));

-- chat_messages: org members can view and insert messages for their events
DROP POLICY IF EXISTS "Members can view org chat messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Members can insert org chat messages" ON public.chat_messages;
CREATE POLICY "Members can view org chat messages" ON public.chat_messages
  FOR SELECT USING (event_id IN (SELECT user_event_ids()));
CREATE POLICY "Members can insert org chat messages" ON public.chat_messages
  FOR INSERT WITH CHECK (event_id IN (SELECT user_event_ids()));
