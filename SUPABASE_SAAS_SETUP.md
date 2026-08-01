# AutoDeal — profils, abonnements et accès

## 1. Appliquer la migration Supabase

Dans Supabase > SQL Editor, exécuter le fichier :

`supabase/migrations/20260802_saas_roles_subscriptions.sql`

Il crée :
- `profiles` : rôle métier (`user`, `samsar`, `dealer`, `admin`)
- `subscriptions` : plan et statut d'abonnement
- les politiques RLS en lecture
- un trigger d'inscription

Les nouveaux comptes obtiennent :
- Particulier -> `role=user`, `plan=free`, `status=active`
- Samsar -> `role=samsar`, `plan=pro`, `status=trialing`, essai 14 jours
- Concessionnaire -> `role=dealer`, `plan=business`, `status=trialing`, essai 14 jours

`admin` n'est jamais sélectionnable à l'inscription.

## 2. Donner le rôle admin à un compte

À faire uniquement manuellement dans le SQL Editor avec l'adresse réelle du compte :

```sql
update public.profiles p
set role = 'admin', updated_at = now()
from auth.users u
where p.user_id = u.id
  and u.email = 'VOTRE_EMAIL';
```

## 3. Modifier manuellement un abonnement pendant les tests

Samsar Pro :

```sql
update public.subscriptions s
set plan='pro', status='active', updated_at=now()
from auth.users u
where s.user_id=u.id and u.email='EMAIL_DU_COMPTE';
```

Concessionnaire Business :

```sql
update public.subscriptions s
set plan='business', status='active', updated_at=now()
from auth.users u
where s.user_id=u.id and u.email='EMAIL_DU_COMPTE';
```

## 4. Paiement

Le module `services/payment_provider.py` est volontairement désactivé.
AutoDeal n'enregistre aucun numéro de carte, CVV ou donnée bancaire.
Lorsqu'un compte marchand sera disponible, `start_checkout()` devra créer une
session chez le prestataire et retourner uniquement son URL de checkout hébergé.

Les prix préparés sont :
- Gratuit : 0 DT
- Pro Samsar : 29 DT/mois ou 299 DT/an
- Business Concessionnaire : 79 DT/mois ou 799 DT/an
- Business+ : 149 DT/mois ou 1499 DT/an

## 5. Sécurité

Ne jamais mettre une service-role/secret key dans le code ou dans les secrets
accessibles au front. La service-role reste réservée aux GitHub Actions/backend.
