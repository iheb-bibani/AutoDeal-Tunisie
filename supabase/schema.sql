-- AutoDeal Tunisie — schéma Supabase canonique
-- À exécuter dans Supabase > SQL Editor pour une installation neuve.
-- Le script est idempotent et inclut Auth/RLS, profils, abonnements, alertes,
-- favoris ainsi que les tables nécessaires au suivi des paiements.

create extension if not exists pgcrypto;

create table if not exists public.notification_settings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text,
  telegram_chat_id text,
  email_enabled boolean not null default true,
  telegram_enabled boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.alerts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null default 'Mon alerte',
  brand text,
  model text,
  budget_max integer,
  max_km integer,
  min_year integer,
  min_gap_pct numeric(5,2) not null default 25,
  active boolean not null default true,
  email_enabled boolean not null default true,
  telegram_enabled boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists alerts_user_id_idx on public.alerts(user_id);
create index if not exists alerts_active_idx on public.alerts(active) where active = true;

create table if not exists public.favorites (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  listing_url text not null,
  listing_snapshot jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique(user_id, listing_url)
);
create index if not exists favorites_user_id_idx on public.favorites(user_id);

create table if not exists public.alert_deliveries (
  id uuid primary key default gen_random_uuid(),
  alert_id uuid not null references public.alerts(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  listing_url text not null,
  channel text not null check (channel in ('email', 'telegram')),
  sent_at timestamptz not null default now(),
  unique(alert_id, listing_url, channel)
);
create index if not exists alert_deliveries_alert_id_idx on public.alert_deliveries(alert_id);

alter table public.notification_settings enable row level security;
alter table public.alerts enable row level security;
alter table public.favorites enable row level security;
alter table public.alert_deliveries enable row level security;

drop policy if exists "own notification settings" on public.notification_settings;
create policy "own notification settings" on public.notification_settings
for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own alerts" on public.alerts;
create policy "own alerts" on public.alerts
for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own favorites" on public.favorites;
create policy "own favorites" on public.favorites
for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own deliveries read" on public.alert_deliveries;
create policy "own deliveries read" on public.alert_deliveries
for select using (auth.uid() = user_id);

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
  status text not null default 'active',
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

-- Colonnes backend de paiement. `provider` est conservé pour compatibilité
-- historique ; `payment_provider` est la colonne utilisée par le code actuel.
alter table public.subscriptions
  add column if not exists payment_provider text,
  add column if not exists last_payment_status text,
  add column if not exists last_payment_at timestamptz,
  add column if not exists next_payment_due_at timestamptz,
  add column if not exists failed_payment_count integer not null default 0;

alter table public.subscriptions drop constraint if exists subscriptions_status_check;
alter table public.subscriptions
  add constraint subscriptions_status_check
  check (status in ('active','trialing','past_due','cancelled','inactive','expired','unpaid'));

alter table public.profiles enable row level security;
alter table public.subscriptions enable row level security;

drop policy if exists "own profile read" on public.profiles;
create policy "own profile read" on public.profiles
for select using (auth.uid() = user_id);

drop policy if exists "own subscription read" on public.subscriptions;
create policy "own subscription read" on public.subscriptions
for select using (auth.uid() = user_id);

-- Aucun UPDATE/INSERT côté client pour les rôles/abonnements : seuls le
-- backend marchand ou un administrateur Supabase peuvent les modifier.

create table if not exists public.payment_transactions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  provider text not null,
  payment_ref text not null unique,
  order_id text,
  plan text not null check (plan in ('pro','business','business_plus')),
  billing_cycle text not null check (billing_cycle in ('monthly','yearly')),
  currency text not null default 'TND',
  amount_tnd numeric(12,3),
  status text not null default 'pending'
    check (status in ('pending','paid','failed','expired','refunded','cancelled')),
  raw_payload jsonb,
  created_at timestamptz not null default now(),
  paid_at timestamptz,
  updated_at timestamptz not null default now()
);

create table if not exists public.subscription_notifications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  kind text not null
    check (kind in ('payment_failed','subscription_expired','payment_success','renewal_due')),
  status text not null default 'pending'
    check (status in ('pending','sent','failed')),
  channel text not null default 'auto'
    check (channel in ('auto','email','telegram')),
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  sent_at timestamptz
);

alter table public.payment_transactions enable row level security;
alter table public.subscription_notifications enable row level security;

drop policy if exists "own payment history read" on public.payment_transactions;
create policy "own payment history read" on public.payment_transactions
for select using (auth.uid() = user_id);

drop policy if exists "own subscription notifications read" on public.subscription_notifications;
create policy "own subscription notifications read" on public.subscription_notifications
for select using (auth.uid() = user_id);

create index if not exists idx_payment_transactions_user
  on public.payment_transactions(user_id, created_at desc);
create index if not exists idx_subscription_notifications_pending
  on public.subscription_notifications(status, created_at);
create index if not exists idx_subscriptions_due
  on public.subscriptions(status, current_period_end);

-- -------------------------------------------------------------------------
-- Création automatique du profil et de l'abonnement lors d'un signup Auth
-- -------------------------------------------------------------------------
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
    initial_plan := 'pro';
    initial_status := 'trialing';
    initial_trial_end := now() + interval '14 days';
  elsif requested_role = 'dealer' then
    initial_plan := 'business';
    initial_status := 'trialing';
    initial_trial_end := now() + interval '14 days';
  else
    initial_plan := 'free';
    initial_status := 'active';
    initial_trial_end := null;
  end if;

  insert into public.subscriptions(
    user_id, plan, status, currency, trial_start, trial_end
  )
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

-- Backfill sécurisé des comptes Auth créés avant l'installation du schéma.
insert into public.profiles(user_id, full_name, role)
select id, coalesce(raw_user_meta_data->>'full_name', raw_user_meta_data->>'name'), 'user'
from auth.users
on conflict (user_id) do nothing;

insert into public.subscriptions(user_id, plan, status, currency)
select id, 'free', 'active', 'TND'
from auth.users
on conflict (user_id) do nothing;
