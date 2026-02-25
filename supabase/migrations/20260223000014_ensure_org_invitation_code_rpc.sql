-- RPC: ensure_organization_invitation_code
-- If the current user's org has no invitation_code, generate one and return it.
-- Used by the admin panel when loading the code for the first time.
CREATE OR REPLACE FUNCTION public.ensure_organization_invitation_code()
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  org_id uuid;
  new_code text;
BEGIN
  IF auth.uid() IS NULL THEN
    RETURN jsonb_build_object('invitation_code', null);
  END IF;

  SELECT organization_id INTO org_id FROM public.user_profiles WHERE id = auth.uid();
  IF org_id IS NULL THEN
    RETURN jsonb_build_object('invitation_code', null);
  END IF;

  -- Generate new code if missing
  UPDATE public.organizations
  SET invitation_code = 'ORB-' || UPPER(SUBSTR(MD5(RANDOM()::TEXT || id::TEXT), 1, 6))
  WHERE id = org_id AND (invitation_code IS NULL OR invitation_code = '')
  RETURNING organizations.invitation_code INTO new_code;

  IF new_code IS NOT NULL THEN
    RETURN jsonb_build_object('invitation_code', new_code);
  END IF;

  -- Already had a code: just return it
  SELECT o.invitation_code INTO new_code FROM public.organizations o WHERE o.id = org_id;
  RETURN jsonb_build_object('invitation_code', new_code);
END;
$$;

GRANT EXECUTE ON FUNCTION public.ensure_organization_invitation_code() TO authenticated;
