# Image de l'API AutoDeal (lecture seule : marché, annonces, deals, estimation).
#   docker build -t autodeal-api .
#   docker run -p 8000:8000 -e AUTODEAL_API_KEY=... autodeal-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY config.py logger.py ./
COPY backend ./backend
COPY core ./core
COPY scrapers/__init__.py scrapers/parsing.py scrapers/liens_vus.py ./scrapers/
COPY data/processed ./data/processed
COPY data/models ./data/models

RUN useradd --create-home autodeal && chown -R autodeal /app
USER autodeal

# IP réelle des clients (limitation de débit) : derrière un reverse proxy,
# remplacer par l'adresse/le CIDR du proxy. Ne jamais mettre "*" : l'en-tête
# X-Forwarded-For deviendrait falsifiable et la limite contournable.
ENV FORWARDED_ALLOW_IPS=127.0.0.1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
