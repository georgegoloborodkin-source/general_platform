-- Add source_doc_ids to chat_messages (for Sources strip in chat), matching remix schema
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

ALTER TABLE public.chat_messages ADD COLUMN IF NOT EXISTS source_doc_ids UUID[];

COMMENT ON COLUMN public.chat_messages.source_doc_ids IS 'Document IDs cited in this message for Sources strip.';
