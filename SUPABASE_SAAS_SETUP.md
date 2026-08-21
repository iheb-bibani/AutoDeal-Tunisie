# AutoDeal — profils, abonnements et accès

Pour une installation neuve, exécute **`supabase/schema.sql`**. Il contient le
schéma complet actuel. Les migrations sous `supabase/migrations/` servent à la
mise à niveau d'anciens projets.

## Rôles

AutoDeal distingue le **rôle métier** de la **formule** :

| Profil | `role` | Formule initiale | Statut initial |
|---|---|---|---|
| Particulier | `user` | `free` | `active` |
| Samsar | `samsar` | `pro` | `trialing` (14 j) |
| Concessionnaire | `dealer` | `business` | `trialing` (14 j) |
| Administrateur | `admin` | accès total | attribution manuelle |

Le rôle `admin` n'est jamais proposé au signup.

## Donner le rôle admin

```sql
update public.profiles p
set role = 'admin', updated_at = now()
from auth.users u
where p.user_id = u.id
  and u.email = 'VOTRE_EMAIL';
```

## Modifier un abonnement pendant les tests

Samsar Pro :

```sql
update public.subscriptions s
set plan='pro', status='active',
    current_period_end=now() + interval '30 days', updated_at=now()
from auth.users u
where s.user_id=u.id and u.email='EMAIL_DU_COMPTE';
```

Concessionnaire Business :

```sql
update public.subscriptions s
set plan='business', status='active',
    current_period_end=now() + interval '30 days', updated_at=now()
from auth.users u
where s.user_id=u.id and u.email='EMAIL_DU_COMPTE';
```

## Paiement

`services/payment_provider.py` sait préparer un checkout **Konnect** lorsque les
secrets marchand sont configurés. AutoDeal n'enregistre jamais PAN/CVV : la
carte est saisie chez le prestataire.

Tarifs préparés :

- Gratuit : 0 DT
- Pro Samsar : 29 DT/mois ou 299 DT/an
- Business Concessionnaire : 79 DT/mois ou 799 DT/an
- Business+ : 149 DT/mois ou 1499 DT/an

Le schéma contient `payment_transactions` et `subscription_notifications`.
`utils/subscription_maintenance.py` expire les périodes payées arrivées à terme
et distribue les notifications backend. L'activation effective d'un paiement
dépend néanmoins d'un compte marchand et d'un flux de confirmation/webhook
réellement configurés : ne considère pas un bouton de checkout comme une preuve
de paiement.

## Sécurité

- Streamlit : `SUPABASE_URL` + `SUPABASE_PUBLISHABLE_KEY` uniquement.
- Backend/GitHub Actions : `SUPABASE_SECRET_KEY` (ou ancien
  `SUPABASE_SERVICE_ROLE_KEY`).
- RLS activée sur les tables utilisateur.
- Les rôles et abonnements ne sont pas modifiables par le client public.
- Les requêtes utilisateur filtrent explicitement sur `user_id` en plus des RLS.
