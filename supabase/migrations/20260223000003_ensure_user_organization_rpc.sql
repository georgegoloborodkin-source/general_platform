-- ensure_user_organization RPC: create org for user if missing and link profile
-- Run in Supabase: Dashboard → SQL Editor → New query → paste and Run
--
-- Frontend calls: supabase.rpc("ensure_user_organization", { org_name, org_slug })

CREATE OR REPLACE FUNCTION public.ensure_user_organization(org_name text, org_slug text)
RETURNS public.organizations
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  existing_org_id uuid;
  created_org public.organizations;
  slug_to_use text := org_slug;
BEGIN
  IF auth.uid() IS NULL THEN
    RAISE EXCEPTION 'not authenticated';
  END IF;

  SELECT organization_id INTO existing_org_id
  FROM public.user_profiles
  WHERE id = auth.uid();

  IF existing_org_id IS NOT NULL THEN
    SELECT * INTO created_org
    FROM public.organizations
    WHERE id = existing_org_id;
    RETURN created_org;
  END IF;

  -- Try insert; on unique violation (slug taken), retry with user-specific suffix
  LOOP
    BEGIN
      INSERT INTO public.organizations (name, slug)
      VALUES (org_name, slug_to_use)
      RETURNING * INTO created_org;
      EXIT;
    EXCEPTION
      WHEN unique_violation THEN
        IF slug_to_use = org_slug THEN
          slug_to_use := org_slug || '-' || substr(replace(auth.uid()::text, '-', ''), 1, 8);
        ELSE
          RAISE;
        END IF;
    END;
  END LOOP;

  UPDATE public.user_profiles
  SET organization_id = created_org.id
  WHERE id = auth.uid();

  RETURN created_org;
END;
$$;

GRANT EXECUTE ON FUNCTION public.ensure_user_organization(text, text) TO authenticated;
