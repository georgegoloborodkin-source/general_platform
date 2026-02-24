-- Add parent_text and optional contextual_header to document_embeddings (used by frontend embed/retrieval)
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

ALTER TABLE public.document_embeddings ADD COLUMN IF NOT EXISTS parent_text TEXT;
ALTER TABLE public.document_embeddings ADD COLUMN IF NOT EXISTS contextual_header TEXT;

COMMENT ON COLUMN public.document_embeddings.parent_text IS 'Parent chunk text for context in RAG retrieval.';
COMMENT ON COLUMN public.document_embeddings.contextual_header IS 'Optional contextual header from chunking.';
