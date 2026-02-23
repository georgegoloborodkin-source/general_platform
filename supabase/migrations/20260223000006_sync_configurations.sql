-- sync_configurations: store Google Drive, Gmail, ClickUp sync settings per org/event
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

CREATE TABLE IF NOT EXISTS public.sync_configurations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  source_type TEXT NOT NULL CHECK (source_type IN ('google_drive', 'gmail', 'clickup')),
  config JSONB NOT NULL DEFAULT '{}',
  sync_frequency TEXT DEFAULT 'hourly' CHECK (sync_frequency IN ('on_login', 'hourly', 'daily')),
  is_active BOOLEAN DEFAULT true,
  created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  last_sync_at TIMESTAMP WITH TIME ZONE,
  last_sync_status TEXT CHECK (last_sync_status IN ('success', 'error', 'pending')),
  last_sync_error TEXT,
  next_sync_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE (organization_id, event_id, source_type)
);

CREATE INDEX IF NOT EXISTS idx_sync_configurations_org_event
  ON public.sync_configurations (organization_id, event_id);
CREATE INDEX IF NOT EXISTS idx_sync_configurations_event_source
  ON public.sync_configurations (event_id, source_type);

ALTER TABLE public.sync_configurations ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Members can view org sync configs" ON public.sync_configurations;
DROP POLICY IF EXISTS "Members can create org sync configs" ON public.sync_configurations;
DROP POLICY IF EXISTS "Members can update org sync configs" ON public.sync_configurations;
DROP POLICY IF EXISTS "Members can delete org sync configs" ON public.sync_configurations;

-- Members can view sync configs for their organization
CREATE POLICY "Members can view org sync configs" ON public.sync_configurations
  FOR SELECT USING (
    organization_id IN (SELECT organization_id FROM public.user_profiles WHERE id = auth.uid())
  );

-- Members can insert sync configs for their organization
CREATE POLICY "Members can create org sync configs" ON public.sync_configurations
  FOR INSERT WITH CHECK (
    organization_id IN (SELECT organization_id FROM public.user_profiles WHERE id = auth.uid())
  );

-- Members can update sync configs for their organization
CREATE POLICY "Members can update org sync configs" ON public.sync_configurations
  FOR UPDATE USING (
    organization_id IN (SELECT organization_id FROM public.user_profiles WHERE id = auth.uid())
  );

-- Members can delete sync configs for their organization
CREATE POLICY "Members can delete org sync configs" ON public.sync_configurations
  FOR DELETE USING (
    organization_id IN (SELECT organization_id FROM public.user_profiles WHERE id = auth.uid())
  );
