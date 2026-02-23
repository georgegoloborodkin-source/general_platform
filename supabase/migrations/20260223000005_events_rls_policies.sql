-- RLS policies for events: allow org members to select, insert, update
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

-- Drop if they exist (safe to re-run)
DROP POLICY IF EXISTS "Members can view org events" ON public.events;
DROP POLICY IF EXISTS "Members can create org events" ON public.events;
DROP POLICY IF EXISTS "Members can update org events" ON public.events;

-- Members can view events of their organization
CREATE POLICY "Members can view org events" ON public.events
  FOR SELECT USING (
    organization_id IN (SELECT organization_id FROM public.user_profiles WHERE id = auth.uid())
  );

-- Members can create events for their organization
CREATE POLICY "Members can create org events" ON public.events
  FOR INSERT WITH CHECK (
    organization_id IN (SELECT organization_id FROM public.user_profiles WHERE id = auth.uid())
  );

-- Members can update events of their organization
CREATE POLICY "Members can update org events" ON public.events
  FOR UPDATE USING (
    organization_id IN (SELECT organization_id FROM public.user_profiles WHERE id = auth.uid())
  );
