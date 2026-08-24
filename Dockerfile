FROM python:3.11-slim

# Éviter les écritures de cache disque inutiles pour Docker
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMBA_NUM_THREADS=1
ENV NUMBA_THREADING_LAYER=workqueue

WORKDIR /app

# Installer les dépendances système minimales pour Tkinter et la compilation
RUN apt-get update && apt-get install -y \
    python3-tk \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Installer les bibliothèques requises avec versions compatibles Numba
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "numpy>=1.26.4,<2.0.0" \
    numba>=0.60.0 \
    nicegui \
    customtkinter \
    scikit-learn \
    pandas \
    scipy \
    xgboost \
    statsmodels \
    codecarbon \
    matplotlib \
    pillow \
    requests \
    openpyxl

# Copier l'ensemble des fichiers du dépôt GitHub
COPY . /app

# Déclarer le port réseau par défaut
EXPOSE 8080

# Lancer directement l'application Web Python
CMD ["python", "app.py"]
