-- Socle minimal imitant Supabase (schéma auth, rôles) pour tester les migrations
-- sur un Postgres nu (CI : job supabase-sql).
create extension if not exists pgcrypto;
create schema if not exists auth;
create table auth.users (id uuid primary key default gen_random_uuid(), email text, email_confirmed_at timestamptz,
  raw_user_meta_data jsonb default '{}'::jsonb);
create or replace function auth.uid() returns uuid language sql stable as $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
do $$ begin create role anon; exception when duplicate_object then null; end $$;
do $$ begin create role authenticated; exception when duplicate_object then null; end $$;
-- Comme Supabase : les nouvelles fonctions du schéma public sont exécutables
-- par anon/authenticated par défaut (c'est ce qui les expose en RPC).
grant usage on schema public to anon, authenticated;
alter default privileges in schema public grant execute on functions to anon, authenticated;
