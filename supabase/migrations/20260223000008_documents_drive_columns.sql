-- Add columns to documents used by Drive sync and app (folder_id, google_drive_*, storage_path, company_entity_id)
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

-- storage_path: optional path/URL for stored file
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS storage_path TEXT;

-- Google Drive sync fields
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS google_drive_file_id TEXT;
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS google_drive_modified_at TIMESTAMPTZ;

-- Optional single folder for quick filtering (also use document_folder_links for many-to-many)
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS folder_id UUID REFERENCES public.source_folders(id) ON DELETE SET NULL;

-- Link document to a primary company entity (kg_entities)
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS company_entity_id UUID REFERENCES public.kg_entities(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_documents_event_google_drive_file_id
  ON public.documents (event_id, google_drive_file_id)
  WHERE google_drive_file_id IS NOT NULL;

COMMENT ON COLUMN public.documents.folder_id IS 'Primary folder for display/filtering; use document_folder_links for full many-to-many.';
