-- Fix ensure_user_organization: when suffixed slug (e.g. g-trader-d7f835a9) already exists,
-- link user to that existing org instead of raising 409. Run in Supabase SQL Editor.

CREATE OR REPLACE FUNCTION public.ensure_user_organization(org_name text, org_slug text)
RETURNS public.organizations
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  existing_org_id uuid;
  created_org public.organizations;
  existing_org public.organizations;
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
          SELECT * INTO existing_org FROM public.organizations WHERE slug = slug_to_use LIMIT 1;
          IF FOUND THEN
            UPDATE public.user_profiles SET organization_id = existing_org.id WHERE id = auth.uid();
            RETURN existing_org;
          END IF;
          slug_to_use := org_slug || '-' || substr(replace(auth.uid()::text, '-', ''), 1, 8) || substr(md5(random()::text), 1, 4);
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
