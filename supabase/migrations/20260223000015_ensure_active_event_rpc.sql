-- RPC: ensure_active_event
-- Returns the current user's org active event, or creates one. Uses SECURITY DEFINER so
-- the insert succeeds even when client-side RLS would fail (e.g. after 429 / token refresh).
CREATE OR REPLACE FUNCTION public.ensure_active_event()
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  org_id uuid;
  ev record;
BEGIN
  IF auth.uid() IS NULL THEN
    RETURN NULL;
  END IF;

  SELECT organization_id INTO org_id FROM public.user_profiles WHERE id = auth.uid();
  IF org_id IS NULL THEN
    RETURN NULL;
  END IF;

  SELECT * INTO ev FROM public.events
  WHERE organization_id = org_id AND status = 'active'
  ORDER BY created_at DESC
  LIMIT 1;

  IF FOUND THEN
    RETURN to_jsonb(ev);
  END IF;

  INSERT INTO public.events (organization_id, name, status)
  VALUES (org_id, 'Main Event', 'active')
  RETURNING * INTO ev;

  RETURN to_jsonb(ev);
END;
$$;

GRANT EXECUTE ON FUNCTION public.ensure_active_event() TO authenticated;
