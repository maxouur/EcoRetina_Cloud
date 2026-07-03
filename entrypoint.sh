#!/bin/bash

# 1. Écran virtuel et gestionnaire de fenêtres
Xvfb :1 -screen 0 1280x800x24 &
sleep 2
fluxbox &
sleep 1

# 2. Serveur VNC graphique (Port 5900)
x11vnc -display :1 -nopw -forever -shared -listen localhost &
sleep 1

# 3. Lancement du pont Web noVNC officiel et mis à jour
# On utilise la commande standard 'novnc_proxy' qui remplace 'launch.sh'
novnc_proxy --vnc localhost:5900 --listen $PORT
