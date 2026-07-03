#!/bin/bash

# En cas d'erreur, on continue
set -e

# 1. Configurer l'affichage sur le Display :0
export DISPLAY=:0

# Nettoyage des verrous système
rm -f /tmp/.X0-lock
rm -f /tmp/.X11-unix/X0

# 2. Lancer l'écran virtuel (Xvfb)
Xvfb :0 -screen 0 1280x800x24 &
sleep 3

# 3. Lancer le gestionnaire de fenêtres Fluxbox
fluxbox &
sleep 2

# 4. Lancer le serveur graphique x11vnc
x11vnc -display :0 -nopw -forever -shared -listen 127.0.0.1 &
sleep 2

# 5. Lancer ton vrai logiciel CustomTkinter en arrière-plan
python GUI_EcoRetina_IAagent.py &
sleep 2

# 6. L'ALCHIMIE NETFLIX/RENDER : Websockify direct
# Il prend le flux vidéo local (5900) et le transforme en site web sur le port Render
# On lui dit de servir directement l'interface graphique incluse dans noVNC
python3 -m websockify --web /usr/share/novnc 0.0.0.0:$PORT 127.0.0.1:5900 --web-path=/websockify
