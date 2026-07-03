#!/bin/bash

# 1. Lancer l'écran virtuel Linux en arrière-plan
Xvfb :1 -screen 0 1280x800x24 &
sleep 2

# 2. Lancer le gestionnaire de fenêtres
fluxbox &
sleep 1

# 3. Lancer le serveur VNC graphique
x11vnc -display :1 -nopw -forever -shared -listen localhost &
sleep 1

# 4. Lancer le pont Web (noVNC) sur le port que Render exige
/usr/share/novnc/utils/launch.sh --vnc localhost:5900 --listen $PORT &
sleep 2

# 5. Lancer ton vrai logiciel CustomTkinter !
python GUI_EcoRetina_IAagent.py
