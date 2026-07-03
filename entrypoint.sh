#!/bin/bash

# 1. Écran virtuel et gestionnaire de fenêtres
Xvfb :1 -screen 0 1280x800x24 &
sleep 2
fluxbox &
sleep 1

# 2. Serveur VNC graphique (Port 5900)
x11vnc -display :1 -nopw -forever -shared -listen localhost &
sleep 1

# 3. noVNC (On le décale sur le port 6080 en interne)
/usr/share/novnc/utils/launch.sh --vnc localhost:5900 --listen 6080 &
sleep 2

# 4. Lancer ton vrai logiciel CustomTkinter !
python GUI_EcoRetina_IAagent.py &

# 5. Le "Bouclier" Render : Un serveur Python sur le port officiel exigé par Render
# Il va servir le dossier noVNC et répondre directement au robot de Render
cd /usr/share/novnc
python3 -m http.server $PORT
