-- Auto-generate invitation_code on new orgs + RPC to join by code
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run

-- Backfill any existing orgs that have no invitation_code
UPDATE public.organizations
SET invitation_code = 'ORB-' || UPPER(SUBSTR(MD5(RANDOM()::TEXT || id::TEXT), 1, 6))
WHERE invitation_code IS NULL;

-- Trigger: auto-generate invitation_code on INSERT if not provided
CREATE OR REPLACE FUNCTION public.set_invitation_code()
RETURNS TRIGGER AS $$
BEGIN
  IF NEW.invitation_code IS NULL OR NEW.invitation_code = '' THEN
    NEW.invitation_code := 'ORB-' || UPPER(SUBSTR(MD5(RANDOM()::TEXT || NEW.id::TEXT), 1, 6));
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_invitation_code ON public.organizations;
CREATE TRIGGER trg_set_invitation_code
  BEFORE INSERT ON public.organizations
  FOR EACH ROW EXECUTE FUNCTION public.set_invitation_code();

-- RPC: join_organization_by_code
-- Team members enter the code → this links them to the org
CREATE OR REPLACE FUNCTION public.join_organization_by_code(code text)
RETURNS public.organizations
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  target_org public.organizations;
  current_org_id uuid;
BEGIN
  IF auth.uid() IS NULL THEN
    RAISE EXCEPTION 'not authenticated';
  END IF;

  -- Already in an org? Return it.
  SELECT organization_id INTO current_org_id FROM public.user_profiles WHERE id = auth.uid();
  IF current_org_id IS NOT NULL THEN
    SELECT * INTO target_org FROM public.organizations WHERE id = current_org_id;
    IF FOUND THEN RETURN target_org; END IF;
  END IF;

  -- Find org by invitation code (case-insensitive)
  SELECT * INTO target_org FROM public.organizations WHERE UPPER(invitation_code) = UPPER(code);
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Invalid invitation code';
  END IF;

  -- Link user to org
  UPDATE public.user_profiles SET organization_id = target_org.id WHERE id = auth.uid();

  RETURN target_org;
END;
$$;

GRANT EXECUTE ON FUNCTION public.join_organization_by_code(text) TO authenticated;
