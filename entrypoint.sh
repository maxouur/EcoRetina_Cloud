#!/bin/bash

# En cas d'erreur, on continue
set -e

# 1. Configurer et forcer l'affichage sur le Display :0
export DISPLAY=:0

# Supprimer d'éventuels verrous d'une session précédente
rm -f /tmp/.X0-lock

# 2. Lancer l'écran virtuel (Xvfb) sur le display :0
Xvfb :0 -screen 0 1280x800x24 &
sleep 3

# 3. Lancer le gestionnaire de fenêtres Fluxbox
fluxbox &
sleep 2

# 4. Lancer le serveur graphique x11vnc sur le display :0
x11vnc -display :0 -nopw -forever -shared &
sleep 2

# 5. Lancer ton vrai logiciel CustomTkinter en arrière-plan
python GUI_EcoRetina_AIagent.py &
sleep 2

# 6. Lancer le proxy de noVNC via son chemin Linux absolu
/usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen $PORT
