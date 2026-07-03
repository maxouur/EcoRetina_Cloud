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
EXPOSE 8080

# Script de lancement : il crée l'écran, lance ton Tkinter, et le streame en HTML5
CMD Xvfb :1 -screen 0 1280x800x24 & \
    sleep 2 && \
    fluxbox & \
    sleep 1 && \
    x11vnc -display :1 -nopw -forever -shared & \
    sleep 1 && \
    /usr/share/novnc/utils/launch.sh --vnc localhost:5900 --listen 8080 & \
    sleep 2 && \
    python GUI_EcoRetina_IAagent.py
