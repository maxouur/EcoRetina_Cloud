FROM python:3.10-slim

# Désactiver les questions interactives de Linux
ENV DEBIAN_FRONTEND=noninteractive

# Étape clé : Installer un mini-bureau Linux virtuel (Xvfb + Fluxbox + noVNC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    python3-tk \
    tk-dev \
    fonts-dejavu \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer tes dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier ton code et tes images
COPY . .

# Configurer l'écran virtuel
ENV DISPLAY=:1
# Rendre le script exécutable
RUN chmod +x entrypoint.sh

# Render attribue un port dynamiquement, pas besoin d'EXPOSE fixe
CMD ["./entrypoint.sh"]
