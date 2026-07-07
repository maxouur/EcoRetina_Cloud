FROM python:3.11-slim

# Éviter les écritures de cache disque inutiles pour Docker
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Installer les dépendances système minimales pour Tkinter et la compilation de XGBoost
RUN apt-get update && apt-get install -y \
    python3-tk \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les bibliothèques requises
RUN pip install --no-cache-dir \
    nicegui \
    customtkinter \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    xgboost \
    statsmodels \
    codecarbon \
    matplotlib \
    pillow \
    openai \
    anthropic \
    groq \
    google-genai \
    requests \
    openpyxl

# Copier l'ensemble des fichiers du dépôt GitHub
COPY . /app

# Déclarer le port réseau par défaut
EXPOSE 8080

# Lancer directement l'application Web Python
CMD ["python", "app.py"]
