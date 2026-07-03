#!/bin/bash

# 1. Écran virtuel et gestionnaire de fenêtres
Xvfb :1 -screen 0 1280x800x24 &
sleep 2
fluxbox &
sleep 1

# 2. Serveur VNC graphique (Port 5900)
x11vnc -display :1 -nopw -forever -shared -listen localhost &
sleep 1

# 3. Lancement direct du pont Web noVNC sur le port exigé par Render
# (Plus besoin de serveur Python secondaire, noVNC gère tout seul)
/usr/share/novnc/utils/launch.sh --vnc localhost:5900 --listen $PORT
