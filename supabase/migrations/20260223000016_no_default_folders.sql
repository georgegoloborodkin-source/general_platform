-- Orbit platform: do not create default folders (Sourcing, Deals, etc.); only user-created folders.
CREATE OR REPLACE FUNCTION public.ensure_default_folders_for_event(p_event_id UUID)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  -- No default folders; users create their own (e.g. "Projects").
  NULL;
END;
$$;
