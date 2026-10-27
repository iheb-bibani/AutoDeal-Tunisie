-- -------------------------------------------------------------------------
-- Essai gratuit : une seule fois par adresse e-mail, après confirmation.
--
-- Avant : n'importe quel signup déclarant account_role = samsar/dealer dans
-- ses métadonnées (appelable directement via l'API Auth avec la clé anon)
-- obtenait 14 jours de Pro/Business. Une nouvelle adresse = un nouvel essai,
-- même jetable et jamais confirmée.
--
-- Maintenant :
--   - le compte est créé en formule gratuite ;
--   - l'essai est accordé quand l'e-mail est CONFIRMÉ (ou immédiatement si la
--     confirmation est désactivée dans Supabase) ;
--   - une adresse normalisée (minuscules, sans "+étiquette", sans points pour
--     Gmail) ne peut ouvrir qu'un seul essai, même après suppression du compte.
-- -------------------------------------------------------------------------

create table if not exists public.trial_claims (
  email_hash text primary key,
  user_id uuid,
  plan text not null,
  claimed_at timestamptz not null default now()
);

-- Table interne : aucune policy, donc aucun accès avec les clés anon/auth.
alter table public.trial_claims enable row level security;

create or replace function public.autodeal_normaliser_email(email text)
returns text
language sql
immutable
as $$
  select case
    when split_part(lower(trim(email)), '@', 2) in ('gmail.com', 'googlemail.com') then
      replace(split_part(split_part(lower(trim(email)), '@', 1), '+', 1), '.', '') || '@gmail.com'
    else
      split_part(split_part(lower(trim(email)), '@', 1), '+', 1) || '@' || split_part(lower(trim(email)), '@', 2)
  end
$$;

create or replace function public.autodeal_accorder_essai(uid uuid, email text, requested_role text)
returns void
language plpgsql
security definer set search_path = public
as $$
declare
  plan_essai text;
  inseres integer;
begin
  if requested_role = 'samsar' then
    plan_essai := 'pro';
  elsif requested_role = 'dealer' then
    plan_essai := 'business';
  else
    return;
  end if;
  if email is null or position('@' in email) = 0 then
    return;
  end if;

  insert into public.trial_claims(email_hash, user_id, plan)
  values (md5(public.autodeal_normaliser_email(email)), uid, plan_essai)
  on conflict (email_hash) do nothing;
  get diagnostics inseres = row_count;
  if inseres = 0 then
    return;  -- essai déjà consommé par cette adresse
  end if;

  -- Uniquement depuis la formule gratuite jamais payée.
  update public.subscriptions
  set plan = plan_essai,
      status = 'trialing',
      trial_start = now(),
      trial_end = now() + interval '14 days',
      updated_at = now()
  where user_id = uid
    and plan = 'free'
    and status = 'active'
    and last_payment_at is null;
end;
$$;

create or replace function public.handle_autodeal_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
declare
  requested_role text;
begin
  requested_role := coalesce(new.raw_user_meta_data->>'account_role', 'user');
  if requested_role not in ('user', 'samsar', 'dealer') then
    requested_role := 'user';
  end if;

  insert into public.profiles(user_id, full_name, role)
  values (
    new.id,
    coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name'),
    requested_role
  )
  on conflict (user_id) do nothing;

  insert into public.subscriptions(user_id, plan, status, currency)
  values (new.id, 'free', 'active', 'TND')
  on conflict (user_id) do nothing;

  -- Confirmation d'e-mail désactivée : l'adresse est déjà "confirmée".
  if new.email_confirmed_at is not null then
    perform public.autodeal_accorder_essai(new.id, new.email, requested_role);
  end if;
  return new;
end;
$$;

create or replace function public.handle_autodeal_email_confirmed()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  if old.email_confirmed_at is null and new.email_confirmed_at is not null then
    perform public.autodeal_accorder_essai(
      new.id,
      new.email,
      (select role from public.profiles where user_id = new.id)
    );
  end if;
  return new;
end;
$$;

-- Fonctions internes : appelées uniquement par les triggers (propriétaire de
-- la fonction). Sans ce REVOKE, PostgREST les exposait en RPC et n'importe
-- quel client (anon/authenticated) pouvait s'accorder un essai.
revoke execute on function public.autodeal_accorder_essai(uuid, text, text) from public, anon, authenticated;
revoke execute on function public.handle_autodeal_new_user() from public, anon, authenticated;
revoke execute on function public.handle_autodeal_email_confirmed() from public, anon, authenticated;

drop trigger if exists on_auth_user_confirmed_autodeal on auth.users;
create trigger on_auth_user_confirmed_autodeal
after update of email_confirmed_at on auth.users
for each row execute procedure public.handle_autodeal_email_confirmed();

-- Les essais déjà ouverts comptent comme consommés.
insert into public.trial_claims(email_hash, user_id, plan, claimed_at)
select md5(public.autodeal_normaliser_email(u.email)), u.id, s.plan, coalesce(s.trial_start, now())
from auth.users u
join public.subscriptions s on s.user_id = u.id
where s.trial_start is not null and u.email is not null
on conflict (email_hash) do nothing;
