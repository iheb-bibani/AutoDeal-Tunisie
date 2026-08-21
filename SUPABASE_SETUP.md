# Supabase — installation AutoDeal Tunisie

Supabase est **optionnel pour la consultation publique** du marché, mais requis
pour l'authentification, les favoris, les alertes personnalisées, les rôles et
les abonnements.

## 1. Initialiser un projet neuf

Dans **Supabase → SQL Editor**, exécute intégralement :

```text
supabase/schema.sql
```

`schema.sql` est désormais le schéma canonique d'installation. Il crée de façon
idempotente :

- `profiles` et `subscriptions` ;
- `favorites`, `alerts`, `notification_settings`, `alert_deliveries` ;
- `payment_transactions`, `subscription_notifications` ;
- les politiques RLS ;
- le trigger `on_auth_user_created_autodeal` ;
- un backfill des comptes déjà présents dans `auth.users`.

Les fichiers de `supabase/migrations/` restent utiles pour comprendre ou mettre
à niveau une ancienne installation, mais **un nouveau projet doit partir de
`schema.sql`**.

## 2. Configurer Streamlit

Dans les secrets de l'application Streamlit :

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."
```

La partie Streamlit utilise uniquement la clé publique avec les RLS. **Ne mets
jamais une secret/service-role key dans le dépôt ni dans un fichier livré au
navigateur.**

## 3. Configurer GitHub Actions / backend

Dans **GitHub → Settings → Secrets and variables → Actions**, ajoute :

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY` — nom recommandé pour la clé backend Supabase
- éventuellement `SUPABASE_SERVICE_ROLE_KEY` pour compatibilité avec une
  ancienne configuration ; le code accepte les deux noms
- `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` pour les alertes globales

Pour l'email :

- `SMTP_HOST`
- `SMTP_PORT` (généralement `587`)
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`

`SMTP_USER` reste accepté par la maintenance d'abonnement pour compatibilité,
mais `SMTP_USERNAME` est le nom canonique.

## 4. Vérifier que l'application pointe vers le bon projet

Après avoir changé `SUPABASE_URL`/`SUPABASE_PUBLISHABLE_KEY`, redémarre
l'application Streamlit puis crée un compte depuis **👤 Mon compte**.

Dans Supabase, vérifie immédiatement :

```text
Authentication → Users
```

Le nouvel utilisateur doit apparaître. Le trigger doit ensuite créer les lignes
correspondantes dans :

```text
profiles
subscriptions
```

Si `Authentication → Users` reste vide, l'application déployée n'utilise pas
les secrets du projet que tu regardes ou n'a pas été redémarrée après leur
modification.

## 5. Donner le rôle admin

Le rôle `admin` n'est jamais sélectionnable à l'inscription. Une fois ton compte
créé, exécute dans SQL Editor :

```sql
update public.profiles p
set role = 'admin', updated_at = now()
from auth.users u
where p.user_id = u.id
  and u.email = 'VOTRE_EMAIL';
```

Puis déconnecte/reconnecte-toi dans AutoDeal. L'admin a accès aux vues Samsar,
Concessionnaire et Admin.

## 6. Confirmation email et mot de passe oublié

Pour la confirmation d'inscription, configure correctement **Authentication →
URL Configuration → Site URL** avec l'URL Streamlit déployée.

Pour le code de récupération :

1. ouvre **Authentication → Email Templates → Reset Password** ;
2. ajoute `{{ .Token }}` au modèle ;
3. enregistre le template.

L'utilisateur pourra ensuite demander le code depuis **Mon compte → Mot de
passe oublié**.

## 7. Flux de données Supabase

```text
Streamlit
  ├─ Supabase Auth
  ├─ profiles / subscriptions (lecture utilisateur)
  ├─ favorites (RLS utilisateur)
  ├─ alerts (RLS utilisateur)
  └─ notification_settings (RLS utilisateur)

GitHub Actions / backend
  ├─ scraping + scoring
  ├─ alertes_bonnes_affaires.csv
  ├─ send_personalized_alerts.py
  └─ subscription_maintenance.py
       ├─ Supabase secret key
       ├─ Telegram Bot API
       └─ SMTP
```

Les données automobiles principales restent dans `data/` et ne sont pas
stockées dans Supabase.
