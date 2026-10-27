-- Vérifie la règle « un essai gratuit par adresse e-mail confirmée ».
-- Chaque contrôle lève une exception (et fait échouer psql) s'il est faux.
\set ON_ERROR_STOP on

create or replace function pg_temp.verifier(cond boolean, message text) returns void language plpgsql as $$
begin
  if not cond then raise exception 'ÉCHEC : %', message; end if;
end $$;

insert into auth.users(id, email, raw_user_meta_data)
values ('00000000-0000-0000-0000-000000000001', 'Ali.Ben+test@Gmail.com', '{"account_role":"samsar"}');
select pg_temp.verifier(
  (select plan = 'free' and status = 'active' from public.subscriptions where user_id = '00000000-0000-0000-0000-000000000001'),
  'un compte non confirmé ne doit pas avoir d''essai');

update auth.users set email_confirmed_at = now() where id = '00000000-0000-0000-0000-000000000001';
select pg_temp.verifier(
  (select plan = 'pro' and status = 'trialing' and trial_end > now() from public.subscriptions
   where user_id = '00000000-0000-0000-0000-000000000001'),
  'la confirmation de l''e-mail doit ouvrir l''essai Pro');

insert into auth.users(id, email, email_confirmed_at, raw_user_meta_data)
values ('00000000-0000-0000-0000-000000000003', 'a.l.i.b.e.n+x@googlemail.com', now(), '{"account_role":"dealer"}');
select pg_temp.verifier(
  (select plan = 'free' from public.subscriptions where user_id = '00000000-0000-0000-0000-000000000003'),
  'une adresse Gmail déguisée ne doit pas ouvrir un second essai');

insert into auth.users(id, email, email_confirmed_at, raw_user_meta_data)
values ('00000000-0000-0000-0000-000000000004', 'garage@exemple.tn', now(), '{"account_role":"dealer"}');
select pg_temp.verifier(
  (select plan = 'business' and status = 'trialing' from public.subscriptions where user_id = '00000000-0000-0000-0000-000000000004'),
  'un concessionnaire confirmé d''emblée doit avoir son essai Business');

insert into auth.users(id, email, email_confirmed_at, raw_user_meta_data)
values ('00000000-0000-0000-0000-000000000005', 'x@y.tn', now(), '{"account_role":"admin"}');
select pg_temp.verifier(
  (select p.role = 'user' and s.plan = 'free' from public.profiles p join public.subscriptions s using (user_id)
   where user_id = '00000000-0000-0000-0000-000000000005'),
  'le rôle admin ne doit jamais pouvoir être demandé à l''inscription');

\echo 'OK : règles d''essai gratuit vérifiées'
