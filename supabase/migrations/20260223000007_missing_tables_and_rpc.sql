-- Missing tables and RPC used by the frontend (decisions, sources, source_folders, kg_*, company_connections, tasks, ensure_default_folders)
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

-- Helper: events the current user can access (org member)
CREATE OR REPLACE FUNCTION public.user_event_ids()
RETURNS SETOF uuid
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT e.id FROM events e
  WHERE e.organization_id IN (SELECT organization_id FROM user_profiles WHERE id = auth.uid())
$$;

-- decisions
CREATE TABLE IF NOT EXISTS public.decisions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  actor_id UUID REFERENCES auth.users(id),
  actor_name TEXT NOT NULL,
  action_type TEXT NOT NULL,
  startup_name TEXT NOT NULL,
  context JSONB,
  confidence_score FLOAT DEFAULT 0,
  outcome TEXT,
  notes TEXT,
  document_id UUID REFERENCES public.documents(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decisions_event_id ON public.decisions (event_id);
ALTER TABLE public.decisions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage decisions" ON public.decisions;
CREATE POLICY "Members can manage decisions" ON public.decisions
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

-- sources
CREATE TABLE IF NOT EXISTS public.sources (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  title TEXT,
  source_type TEXT NOT NULL DEFAULT 'upload',
  external_url TEXT,
  storage_path TEXT,
  tags TEXT[],
  notes TEXT,
  status TEXT DEFAULT 'active',
  created_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sources_event_id ON public.sources (event_id);
ALTER TABLE public.sources ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage sources" ON public.sources;
CREATE POLICY "Members can manage sources" ON public.sources
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

-- source_folders
CREATE TABLE IF NOT EXISTS public.source_folders (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  created_by UUID REFERENCES auth.users(id),
  category TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_source_folders_event_id ON public.source_folders (event_id);
ALTER TABLE public.source_folders ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage source_folders" ON public.source_folders;
CREATE POLICY "Members can manage source_folders" ON public.source_folders
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

-- document_folder_links (for folder-document association)
CREATE TABLE IF NOT EXISTS public.document_folder_links (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
  folder_id UUID NOT NULL REFERENCES public.source_folders(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE (document_id, folder_id)
);

CREATE INDEX IF NOT EXISTS idx_document_folder_links_folder_id ON public.document_folder_links (folder_id);
ALTER TABLE public.document_folder_links ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage document_folder_links" ON public.document_folder_links;
CREATE POLICY "Members can manage document_folder_links" ON public.document_folder_links
  FOR ALL USING (
    folder_id IN (SELECT sf.id FROM source_folders sf WHERE sf.event_id IN (SELECT user_event_ids()))
  );

-- kg_entities
CREATE TABLE IF NOT EXISTS public.kg_entities (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  entity_type TEXT NOT NULL,
  name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  properties JSONB DEFAULT '{}',
  source_document_id UUID REFERENCES public.documents(id),
  confidence FLOAT DEFAULT 0.5,
  created_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_event_id ON public.kg_entities (event_id);
CREATE INDEX IF NOT EXISTS idx_kg_entities_normalized_name ON public.kg_entities (event_id, normalized_name, entity_type);
ALTER TABLE public.kg_entities ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage kg_entities" ON public.kg_entities;
CREATE POLICY "Members can manage kg_entities" ON public.kg_entities
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

-- kg_edges
CREATE TABLE IF NOT EXISTS public.kg_edges (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  source_entity_id UUID NOT NULL REFERENCES public.kg_entities(id) ON DELETE CASCADE,
  target_entity_id UUID NOT NULL REFERENCES public.kg_entities(id) ON DELETE CASCADE,
  relation_type TEXT NOT NULL,
  properties JSONB DEFAULT '{}',
  source_document_id UUID REFERENCES public.documents(id),
  confidence FLOAT DEFAULT 0.5,
  created_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  review_status TEXT,
  reviewed_by UUID REFERENCES auth.users(id),
  reviewed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_kg_edges_event_id ON public.kg_edges (event_id);
ALTER TABLE public.kg_edges ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage kg_edges" ON public.kg_edges;
CREATE POLICY "Members can manage kg_edges" ON public.kg_edges
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

-- company_connections
CREATE TABLE IF NOT EXISTS public.company_connections (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  created_by UUID REFERENCES auth.users(id),
  source_company_name TEXT NOT NULL,
  target_company_name TEXT NOT NULL,
  source_document_id UUID REFERENCES public.documents(id),
  target_document_id UUID REFERENCES public.documents(id),
  connection_type TEXT NOT NULL,
  connection_status TEXT NOT NULL,
  ai_reasoning TEXT,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_company_connections_event_id ON public.company_connections (event_id);
ALTER TABLE public.company_connections ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage company_connections" ON public.company_connections;
CREATE POLICY "Members can manage company_connections" ON public.company_connections
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

-- tasks
CREATE TABLE IF NOT EXISTS public.tasks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  assignee_user_id UUID REFERENCES auth.users(id),
  title TEXT NOT NULL,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'not_started' CHECK (status IN ('not_started', 'in_progress', 'completed', 'cancelled')),
  start_date DATE,
  deadline TIMESTAMP WITH TIME ZONE,
  created_by UUID NOT NULL REFERENCES auth.users(id),
  status_note TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tasks_event_id ON public.tasks (event_id);
ALTER TABLE public.tasks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage tasks" ON public.tasks;
CREATE POLICY "Members can manage tasks" ON public.tasks
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

-- RPC: ensure default source_folders for an event (SECURITY DEFINER so it can insert despite RLS)
CREATE OR REPLACE FUNCTION public.ensure_default_folders_for_event(p_event_id UUID)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  r RECORD;
  default_folders JSONB := '[
    {"name": "Companies", "category": "Companies"},
    {"name": "Partners", "category": "Partners"},
    {"name": "Deals", "category": "Sourcing"},
    {"name": "Market Research", "category": "Sourcing"},
    {"name": "Due Diligence", "category": "Companies"},
    {"name": "BD", "category": "BD"}
  ]'::jsonb;
  item JSONB;
  existing_names TEXT[];
BEGIN
  SELECT array_agg(LOWER(name)) INTO existing_names
  FROM source_folders WHERE event_id = p_event_id;

  FOR item IN SELECT * FROM jsonb_array_elements(default_folders)
  LOOP
    IF existing_names IS NULL OR NOT (LOWER(item->>'name') = ANY(existing_names)) THEN
      INSERT INTO source_folders (event_id, name, category)
      VALUES (p_event_id, item->>'name', item->>'category');
    END IF;
  END LOOP;
END;
$$;
