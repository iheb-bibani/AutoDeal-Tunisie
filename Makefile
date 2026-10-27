.PHONY: install test lint fix pipeline app api docker sql-test

install:            ## Dépendances complètes (pipeline, app, API, tests)
	pip install -r requirements.txt ruff pre-commit

test:               ## Tests Python
	python -m pytest -q

lint:               ## Lint bloquant (mêmes règles que la CI)
	ruff check .

fix:
	ruff check . --fix

pipeline:           ## Pipeline complet (scraping + traitement), comme la nuit
	python main.py

traitement:         ## Traitement seul, sur les CSV bruts existants
	python core/merging_files.py && python core/nettoyer_base.py && \
	python core/enrichir_base_avance.py && python core/modele_prediction.py && \
	python core/detect_deals.py && python core/suivi_annonces.py && python core/quality_gate.py

app:                ## Application Streamlit
	streamlit run app.py

api:                ## API FastAPI en local (docs : http://127.0.0.1:8000/docs)
	uvicorn backend.main:app --reload --port 8000

docker:             ## Image de l'API
	docker build -t autodeal-api .
