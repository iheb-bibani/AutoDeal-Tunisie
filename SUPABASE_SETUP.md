# Supabase + alertes personnalisées — AutoDeal Tunisie

## 1. Créer le projet Supabase

Crée un projet Supabase puis ouvre **SQL Editor** et exécute intégralement `supabase/schema.sql`.
Le script crée `alerts`, `favorites`, `notification_settings` et `alert_deliveries`, puis active la Row Level Security (RLS).

## 2. Configurer Streamlit Cloud

Dans les secrets de l'application Streamlit, ajoute :

```toml
SUPABASE_URL = "https://...supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."
```

La publishable key peut être utilisée par l'application avec les politiques RLS. **Ne mets jamais la secret/service-role key dans les secrets accessibles au code client ou dans le dépôt.**

## 3. Configurer GitHub Actions

Dans GitHub > Settings > Secrets and variables > Actions, ajoute :

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY` : secret/service-role key Supabase, réservée au workflow serveur
- `TELEGRAM_TOKEN` : token du bot AutoDeal (déjà utilisé par l'alerte globale)

Pour les emails, ajoute aussi :

- `SMTP_HOST`
- `SMTP_PORT` (souvent `587`)
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`

Si les secrets SMTP sont absents, le script continue et ignore simplement le canal email.

## 4. Authentification

L'inscription email/mot de passe utilise Supabase Auth. Par défaut, les projets Supabase hébergés demandent généralement la confirmation de l'adresse email. Configure le **Site URL** et les redirect URLs dans Supabase Auth si tu conserves la confirmation email.

## 5. Utilisation dans AutoDeal

1. Ouvrir **👤 Mon compte** et créer un compte.
2. Enregistrer l'email de notification et, si souhaité, le Chat ID Telegram.
3. Depuis une annonce, cliquer **♡ Favori** pour la retrouver plus tard.
4. Ouvrir **🔔 Alertes**, choisir marque/modèle/budget/km/année/écart marché et les canaux.
5. Chaque nuit, après le scraping, `utils/send_personalized_alerts.py` compare les bonnes affaires aux alertes actives.
6. `alert_deliveries` empêche de renvoyer le même lien pour la même alerte et le même canal.

## Architecture

```text
Streamlit
  ├─ Supabase Auth
  ├─ favorites (RLS utilisateur)
  ├─ alerts (RLS utilisateur)
  └─ notification_settings (RLS utilisateur)

GitHub Actions / pipeline nocturne
  ├─ scraping + scoring
  ├─ alertes_bonnes_affaires.csv
  └─ send_personalized_alerts.py
       ├─ Supabase secret key
       ├─ Telegram Bot API
       ├─ SMTP email
       └─ alert_deliveries (déduplication)
```

## Mot de passe oublié — configuration du code recovery

L'interface AutoDeal utilise un **code de récupération** afin de rester compatible avec Streamlit sans JavaScript côté navigateur.
Dans Supabase :

1. Ouvrez **Authentication > Email Templates > Reset Password**.
2. Dans le modèle d'email, affichez le code Supabase avec `{{ .Token }}`.
3. Exemple de contenu : `Votre code de récupération AutoDeal est : {{ .Token }}`.
4. Enregistrez le modèle.

L'utilisateur pourra ensuite aller dans **Mon compte > Mot de passe oublié**, demander un code, le saisir et choisir un nouveau mot de passe.

Pour la confirmation d'inscription, conservez le template de confirmation Supabase standard et vérifiez que **Authentication > URL Configuration > Site URL** pointe vers l'URL Streamlit déployée.
