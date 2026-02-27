-- company_kpis: extracted KPIs/metrics per company (used by entity extraction from .xlsx and other docs)
-- Fixes 404 "Could not find the table 'public.company_kpis'" during extraction

CREATE TABLE IF NOT EXISTS public.company_kpis (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_id UUID NOT NULL REFERENCES public.events(id) ON DELETE CASCADE,
  company_name TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  value TEXT,
  unit TEXT,
  period TEXT,
  metric_category TEXT,
  confidence FLOAT DEFAULT 0.5,
  source_document_id UUID REFERENCES public.documents(id) ON DELETE SET NULL,
  extraction_method TEXT DEFAULT 'claude_extraction',
  created_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_company_kpis_event_id ON public.company_kpis (event_id);
CREATE INDEX IF NOT EXISTS idx_company_kpis_event_company_metric_period ON public.company_kpis (event_id, company_name, metric_name, period);

ALTER TABLE public.company_kpis ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Members can manage company_kpis" ON public.company_kpis;
CREATE POLICY "Members can manage company_kpis" ON public.company_kpis
  FOR ALL USING (event_id IN (SELECT user_event_ids()));

COMMENT ON TABLE public.company_kpis IS 'Structured KPIs/metrics extracted from documents (e.g. MRR, headcount) for RAG and company cards.';
