-- Add events.date so existing frontend bundles that order by "date" work
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

ALTER TABLE public.events
  ADD COLUMN IF NOT EXISTS date TIMESTAMP WITH TIME ZONE DEFAULT NOW();

UPDATE public.events
  SET date = COALESCE(created_at, NOW())
  WHERE date IS NULL;
