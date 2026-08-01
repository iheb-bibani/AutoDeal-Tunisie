-- Synchronisation paiement <-> accès AutoDeal

alter table public.subscriptions
  add column if not exists payment_provider text,
  add column if not exists last_payment_status text,
  add column if not exists last_payment_at timestamptz,
  add column if not exists next_payment_due_at timestamptz,
  add column if not exists failed_payment_count integer not null default 0;

-- L'ancien schéma n'autorisait pas expired/unpaid.
alter table public.subscriptions drop constraint if exists subscriptions_status_check;
alter table public.subscriptions
  add constraint subscriptions_status_check
  check (status in ('active','trialing','past_due','cancelled','inactive','expired','unpaid'));

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
  status text not null default 'pending' check (status in ('pending','paid','failed','expired','refunded','cancelled')),
  raw_payload jsonb,
  created_at timestamptz not null default now(),
  paid_at timestamptz,
  updated_at timestamptz not null default now()
);

create table if not exists public.subscription_notifications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  kind text not null check (kind in ('payment_failed','subscription_expired','payment_success','renewal_due')),
  status text not null default 'pending' check (status in ('pending','sent','failed')),
  channel text not null default 'auto' check (channel in ('auto','email','telegram')),
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  sent_at timestamptz
);

alter table public.payment_transactions enable row level security;
alter table public.subscription_notifications enable row level security;

-- L'utilisateur peut consulter son propre historique, jamais le modifier.
drop policy if exists "own payment history read" on public.payment_transactions;
create policy "own payment history read" on public.payment_transactions
for select using (auth.uid() = user_id);

drop policy if exists "own subscription notifications read" on public.subscription_notifications;
create policy "own subscription notifications read" on public.subscription_notifications
for select using (auth.uid() = user_id);

create index if not exists idx_payment_transactions_user on public.payment_transactions(user_id, created_at desc);
create index if not exists idx_subscription_notifications_pending on public.subscription_notifications(status, created_at);
create index if not exists idx_subscriptions_due on public.subscriptions(status, current_period_end);
