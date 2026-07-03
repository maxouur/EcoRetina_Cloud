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

# 4. Lancer le serveur graphique x11vnc sur le port 5900
x11vnc -display :0 -nopw -forever -shared -listen 127.0.0.1 &
sleep 2

# 5. Lancer le proxy noVNC en arrière-plan sur le port 6080
/usr/share/novnc/utils/novnc_proxy --vnc 127.0.0.1:5900 --listen 6080 &
sleep 2

# 6. Lancer ton logiciel CustomTkinter en arrière-plan
python GUI_EcoRetina_AIagent.py &
sleep 2

# 7. Le Bouclier Render : Serveur Web Python officiel sur le port demandé
# Il va servir les fichiers de noVNC directement à Render
cd /usr/share/novnc
python3 -m http.server $PORT
