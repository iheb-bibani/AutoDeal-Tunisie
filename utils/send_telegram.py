name: Scraping quotidien

on:
  schedule:
    # Minuit heure tunisienne (UTC+1, pas de changement d'heure)
    - cron: '0 23 * * *'
  workflow_dispatch:

# Un scraping dure plusieurs heures : sans verrou, le run du lendemain
# demarrerait par-dessus celui de la veille.
concurrency:
  group: scraping
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  scraping:
    runs-on: ubuntu-latest
    timeout-minutes: 350

    steps:
      # fetch-depth: 0 -> sans historique complet, le rebase de la boucle de
      # publication ne peut pas se raccorder au distant et le push est rejete.
      - name: Recuperer le depot
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Installer Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip

      - name: Installer les dependances
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          python -m playwright install --with-deps chromium
          python -m patchright install chromium

      # Patchright tourne en non-headless avec channel="chrome" pour passer
      # Cloudflare Turnstile : il faut un serveur X virtuel.
      - name: Installer xvfb et Chrome stable
        run: |
          sudo apt-get update
          sudo apt-get install -y xvfb
          python -m patchright install chrome || true

      - name: Lancer le pipeline
        id: pipeline
        run: xvfb-run -a python main.py

      # Sans ce controle, un pipeline interrompu laisse les anciens fichiers en
      # place : le commit ne montre aucun changement et tout parait normal.
      - name: Verifier les donnees produites
        id: verif
        run: |
          FICHIER=data/processed/tunisia-cars-scored.csv
          if [ ! -s "$FICHIER" ]; then
            echo "::error::$FICHIER absent ou vide"
            exit 1
          fi
          LIGNES=$(($(wc -l < "$FICHIER") - 1))
          echo "Annonces scorees : $LIGNES"
          if [ "$LIGNES" -lt 100 ]; then
            echo "::error::Seulement $LIGNES annonces scorees (minimum 100)"
            exit 1
          fi

      # Une alerte non partie n'invalide pas la collecte : cette etape ne doit
      # pas faire echouer le job.
      - name: Envoyer les alertes Telegram
        continue-on-error: true
        env:
          TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python utils/send_telegram.py

      - name: Enregistrer les nouvelles donnees dans le depot
        if: always()
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

          # Publication differenciee : data/raw est append-only et jamais
          # incoherent, il part meme en cas d'echec pour ne pas jeter des heures
          # de collecte. data/processed et data/models ne partent que si tout a
          # reussi -- et sont restaures avant le rebase, car des fichiers
          # modifies non stages rendraient l'arbre sale et git rebase refuserait
          # de tourner.
          if [ "${{ steps.verif.outcome }}" = "success" ]; then
            echo "Pipeline complet : publication de raw + processed + models"
            git add data/raw data/processed data/models
          else
            echo "Pipeline en echec : publication de data/raw uniquement"
            git checkout -- data/processed data/models 2>/dev/null || true
            git clean -fd data/processed data/models || true
            git add data/raw
          fi

          if git diff --cached --quiet; then
            echo "Rien a publier"
            exit 0
          fi

          git commit -m "Scraping automatique $(date -u '+%Y-%m-%d %H:%M UTC')"

          # Pendant un rebase les roles sont inverses par rapport a un merge :
          # 'ours' designe la branche rejouee (le distant), 'theirs' les commits
          # rejoues (les donnees scrapees). -X ours jetterait le scraping.
          for TENTATIVE in 1 2 3 4 5; do
            echo "Publication, tentative $TENTATIVE/5..."
            git fetch origin main
            if git rebase -X theirs origin/main; then
              if git push origin HEAD:main; then
                echo "Publication reussie"
                exit 0
              fi
              echo "Push rejete, nouvelle tentative"
            else
              echo "Rebase impossible, abandon de la tentative"
              git rebase --abort || true
            fi
            sleep $(( RANDOM % 10 + 5 ))
          done

          echo "::error::Echec de publication apres 5 tentatives"
          exit 1
