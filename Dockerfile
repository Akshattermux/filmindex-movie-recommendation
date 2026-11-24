FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --disable-pip-version-check --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 film
COPY --chown=film:film .streamlit/config.toml .streamlit/config.toml
COPY --chown=film:film filmindex/ filmindex/
COPY --chown=film:film app.py .
COPY --chown=film:film assets/ assets/
COPY --chown=film:film artifacts/catalog.pkl artifacts/catalog.pkl
USER film
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]
