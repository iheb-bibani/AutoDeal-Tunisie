-- -------------------------------------------------------------------------
-- SaaS : profils métier + abonnements
-- -------------------------------------------------------------------------
create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  full_name text,
  role text not null default 'user' check (role in ('user', 'samsar', 'dealer', 'admin')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.subscriptions (
  user_id uuid primary key references auth.users(id) on delete cascade,
  plan text not null default 'free' check (plan in ('free', 'pro', 'business', 'business_plus')),
  status text not null default 'active' check (status in ('active', 'trialing', 'past_due', 'cancelled', 'inactive')),
  billing_cycle text check (billing_cycle in ('monthly', 'yearly')),
  currency text not null default 'TND',
  provider text,
  provider_customer_id text,
  provider_subscription_id text,
  trial_start timestamptz,
  trial_end timestamptz,
  current_period_start timestamptz,
  current_period_end timestamptz,
  cancel_at_period_end boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
alter table public.subscriptions enable row level security;

drop policy if exists "own profile read" on public.profiles;
create policy "own profile read" on public.profiles
for select using (auth.uid() = user_id);

drop policy if exists "own subscription read" on public.subscriptions;
create policy "own subscription read" on public.subscriptions
for select using (auth.uid() = user_id);

-- Pas de policy UPDATE/INSERT côté client : rôle et abonnement ne sont pas
-- modifiables depuis Streamlit. Les changements sont effectués par le backend
-- marchand ou manuellement par l'administrateur avec la service-role key.

create or replace function public.handle_autodeal_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
declare
  requested_role text;
  initial_plan text;
  initial_status text;
  initial_trial_end timestamptz;
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

  if requested_role = 'samsar' then
    initial_plan := 'pro'; initial_status := 'trialing'; initial_trial_end := now() + interval '14 days';
  elsif requested_role = 'dealer' then
    initial_plan := 'business'; initial_status := 'trialing'; initial_trial_end := now() + interval '14 days';
  else
    initial_plan := 'free'; initial_status := 'active'; initial_trial_end := null;
  end if;

  insert into public.subscriptions(user_id, plan, status, currency, trial_start, trial_end)
  values (
    new.id, initial_plan, initial_status, 'TND',
    case when initial_status='trialing' then now() else null end,
    initial_trial_end
  )
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created_autodeal on auth.users;
create trigger on_auth_user_created_autodeal
after insert on auth.users
for each row execute procedure public.handle_autodeal_new_user();

-- Backfill sécurisé des comptes existants : particulier + gratuit par défaut.
insert into public.profiles(user_id, full_name, role)
select id, coalesce(raw_user_meta_data->>'full_name', raw_user_meta_data->>'name'), 'user'
from auth.users
on conflict (user_id) do nothing;

insert into public.subscriptions(user_id, plan, status, currency)
select id, 'free', 'active', 'TND'
from auth.users
on conflict (user_id) do nothing;
