#!/bin/bash

# En cas d'erreur, on continue
set -e

# 1. Forcer l'affichage interne
export DISPLAY=:0

# Nettoyage des processus fantômes et verrous
killall Xvfb x11vnc websockify novnc_proxy python3 python 2>/dev/null || true
rm -f /tmp/.X0-lock
rm -f /tmp/.X11-unix/X0

# 2. Lancer l'écran virtuel (Xvfb)
Xvfb :0 -screen 0 1280x800x24 &
sleep 3

# 3. Lancer le gestionnaire de fenêtres Fluxbox
fluxbox &
sleep 2

# 4. LE REGLAGE CRUCIAL : x11vnc écoute STRICTEMENT en local sur le port 5900
# On lui interdit d'écouter sur le port global de Render ($PORT)
x11vnc -display :0 -nopw -forever -shared -listen 127.0.0.1 -rfbport 5900 &
sleep 2

# 5. Lancer ton logiciel CustomTkinter en arrière-plan
python GUI_EcoRetina_AIagent.py &
sleep 2

# 6. LE SEUL MAÎTRE DU PORT DE RENDER : Websockify
# C'est lui et lui seul qui prend le port public ($PORT) pour répondre proprement à Render
# Il va transformer le flux local 5900 en page Web
python3 -m websockify --web /usr/share/novnc 0.0.0.0:$PORT 127.0.0.1:5900
