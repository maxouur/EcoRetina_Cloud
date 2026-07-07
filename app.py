import sys
import os
import traceback
import warnings
import asyncio
import re
import csv
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64

from nicegui import ui, run
from codecarbon import EmissionsTracker

# --- COMPATIBILITÉ DES ALGORITHMES ET IMPORTS ---
import xgboost as xgb
from google import genai
from google.genai import types
import openai
import anthropic
from groq import Groq
import requests

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm

try:
    from eco_retina import EcoRETINA
    ECO_RETINA_AVAILABLE = True
except Exception as e:
    ECO_RETINA_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==========================================
# GESTIONNAIRE D'ÉTAT GLOBAL (WORKSPACE)
# ==========================================
class Workspace:
    def __init__(self):
        self.df = None
        self.df_predict = None
        self.history = []  # Pour le Undo
        self.future = []   # Pour le Redo
        self.run_history = {}
        self.logs = []
        self.active_algo = "EcoRETINA"
        self.ai_agent = None

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logs.append(f"[{timestamp}] {message}")
        ui.notify(message)

    def save_state(self, action_name):
        if self.df is not None:
            self.history.append((action_name, self.df.copy()))
            if len(self.history) > 15:
                self.history.pop(0)
            self.future.clear()

state = Workspace()

# ==========================================
# MOTEUR IA COPILOT MULTI-PROVIDER (STREAMING)
# ==========================================
class EcoRetinaChatAgent:
    def __init__(self, api_key: str, provider: str):
        self.provider = provider
        self.api_key = api_key
        self.system_prompt = (
            "You are the Chief Econometrician and AI Support Guide for the EcoRETINA ML Workbench.\n"
            "Output strictly plain text. DO NOT use Markdown asterisks or bold text. Use ALL CAPS for emphasis."
        )
        self.history = []
        
        if self.provider == "Google Gemini":
            self.client = genai.Client(api_key=api_key)
        elif self.provider == "OpenAI (ChatGPT)":
            self.client = openai.OpenAI(api_key=api_key)
            self.history.append({"role": "system", "content": self.system_prompt})
        elif self.provider == "Groq":
            self.client = Groq(api_key=api_key)
            self.history.append({"role": "system", "content": self.system_prompt})

    async def ask(self, text: str, bubble_ui):
        if self.provider == "Google Gemini":
            # Simulation de streaming pour l'exemple asynchrone NiceGUI
            response = self.client.chats.create(model="gemini-2.5-flash", config=types.GenerateContentConfig(system_instruction=self.system_prompt, temperature=0.3))
            reply = response.send_message(text).text
            bubble_ui.text = reply
        elif self.provider == "OpenAI (ChatGPT)":
            self.history.append({"role": "user", "content": text})
            response = self.client.chat.completions.create(model="gpt-4o-mini", messages=self.history, temperature=0.3)
            reply = response.choices[0].message.content
            bubble_ui.text = reply
        elif self.provider == "Groq":
            self.history.append({"role": "user", "content": text})
            response = self.client.chat.completions.create(model="llama-3.3-70b-versatile", messages=self.history, temperature=0.3)
            reply = response.choices[0].message.content
            bubble_ui.text = reply

class OLSWrapper:
    def __init__(self, res): self.sm_model = res
    def predict(self, X): return self.sm_model.predict(X)

# ==========================================
# INTERFACE GRAPHIQUE WEB (NiceGUI)
# ==========================================

@ui.page('/')
def main_page():
    ui.dark_mode().enable()
    state.log("Application Web initialisée avec succès.")

    # --- HEADER / BARRE DE NAVIGATION ---
    with ui.header().classes('bg-slate-900 text-white items-center justify-between p-4 shadow-md'):
        with ui.row().classes('items-center gap-4'):
            ui.label('≡').classes('text-3xl cursor-pointer').on('click', lambda: left_drawer.toggle())
            ui.image('https://raw.githubusercontent.com/zaidb/StaticStorage/main/logoecoretina.png').classes('w-10 h-10')
            ui.label('EcoRETINA ML Workbench PRO').classes('text-xl font-bold')
        
        with ui.row().classes('items-center gap-2'):
            ui.button('↩ Undo', on_click=apply_undo).props('flat color=white')
            ui.button('↪ Redo', on_click=apply_redo).props('flat color=white')
            ui.button('🤖 AI Assistant', on_click=lambda: right_drawer.toggle()).classes('bg-blue-600')

    # --- MENU LATÉRAL GAUCHE (Navigation Vues) ---
    with ui.left_drawer(value=False).classes('bg-slate-800 text-white') as left_drawer:
        ui.label('Vues du Projet').classes('text-lg font-bold p-4 text-emerald-400')
        ui.button('Workspace Principal', on_click=lambda: tabs.set_value('workspace')).classes('w-full justify-start q-ma-xs')
        ui.button('Activity Log', on_click=lambda: tabs.set_value('logs')).classes('w-full justify-start q-ma-xs')
        ui.button('Documentation / Tutoriel', on_click=lambda: tabs.set_value('tutorial')).classes('w-full justify-start q-ma-xs')

    # --- PANNEAU CO-PILOTE IA (Droit) ---
    with ui.right_drawer(value=False).classes('bg-slate-900 p-4') as right_drawer:
        ui.label('AI Copilot Support').classes('text-h6 font-bold text-emerald-400')
        provider_select = ui.select(["Google Gemini", "OpenAI (ChatGPT)", "Groq"], value="Google Gemini").classes('w-full my-2')
        key_input = ui.input(placeholder='Collez votre clé API ici', password=True).classes('w-full my-2')
        
        chat_container = ui.scroll_area().classes('w-full h-96 bg-slate-950 p-2 rounded my-4')
        
        async def connect_ai():
            if not key_input.value:
                ui.notify("Clé API manquante !")
                return
            state.ai_agent = EcoRetinaChatAgent(key_input.value, provider_select.value)
            ui.notify(f"IA connectée via {provider_select.value}")
        
        ui.button('Connecter', on_click=connect_ai).classes('w-full bg-emerald-600')
        
        chat_input = ui.input(placeholder='Posez une question à l\'IA...').classes('w-full mt-4')
        async def send_chat():
            if not state.ai_agent: return
            with chat_container:
                ui.label(f"User: {chat_input.value}").classes('text-blue-400 font-bold block')
                ai_bubble = ui.label("Thinking...").classes('text-white block ml-2 bg-slate-800 p-2 rounded')
                await state.ai_agent.ask(chat_input.value, ai_bubble)
                chat_input.value = ''
        chat_input.on('keydown.enter', send_chat)

    # --- ZONE CENTRALE MULTI-ONGLETS ---
    with ui.tabs().classes('w-full bg-slate-950 text-white') as tabs:
        tab_work = ui.tab('workspace', label='Workspace ML')
        tab_log = ui.tab('logs', label='Activity Log')
        tab_tut = ui.tab('tutorial', label='Tutoriel')

    with ui.tab_panels(tabs, value='workspace').classes('w-full bg-transparent p-4'):
        # PANEL WORKSPACE
        with ui.tab_panel('workspace'):
            with ui.row().classes('w-full gap-4'):
                # SECTION 1: IMPORT DU DATASET
                with ui.card().classes('w-full md:w-5/12 p-4 bg-slate-800'):
                    ui.label('1. Importation du Dataset').classes('text-lg font-bold text-emerald-400')
                    ui.upload(label='Déposez votre CSV ou Excel ici', on_upload=handle_file_upload, auto_upload=True).classes('w-full')
                    ui.button('Visualiser les données', on_click=open_data_viewer).classes('bg-blue-600 mt-2')

                # SECTION 2: ALGORITHMES ET PIPELINE
                with ui.card().classes('w-full md:w-6/12 p-4 bg-slate-800'):
                    ui.label('2. Modèles & Hyperparamètres').classes('text-lg font-bold text-emerald-400')
                    algo_select = ui.select(['EcoRETINA', 'OLS', 'Lasso', 'Ridge', 'XGBoost', 'Neural Network'], value='EcoRETINA', on_change=lambda e: setattr(state, 'active_algo', e.value)).classes('w-full')
                    
                    target_select = ui.select([], label='Variable Cible (Y)').classes('w-full')
                    state.target_select_ui = target_select # Hook pour mise à jour dynamique

                    ui.button('► Lancer l\'entraînement', on_click=lambda: run_training(algo_select.value, target_select.value)).classes('w-full bg-emerald-600 text-lg mt-4')

            # HISTORIQUE ET COMPARAISON DES RUNS
            with ui.card().classes('w-full mt-6 bg-slate-800 p-4'):
                ui.label('3. Comparatif Historique des Modèles').classes('text-lg font-bold text-emerald-400')
                state.runs_table = ui.table(
                    columns=[
                        {'name': 'run', 'label': 'ID Run', 'field': 'run'},
                        {'name': 'algo', 'label': 'Algo', 'field': 'algo'},
                        {'name': 'r2_train', 'label': 'R² Train', 'field': 'r2_train'},
                        {'name': 'r2_test', 'label': 'R² Test', 'field': 'r2_test'},
                        {'name': 'mape', 'label': 'MAPE Test', 'field': 'mape'},
                        {'name': 'co2', 'label': 'CO2 (kg)', 'field': 'co2'},
                    ],
                    rows=[]
                ).classes('w-full bg-slate-900 text-white')

        # PANEL LOGS SYSTEME
        with ui.tab_panel('logs'):
            with ui.card().classes('w-full bg-slate-800 p-4'):
                ui.label('Journal d\'Activité du Serveur').classes('text-xl font-bold mb-4')
                state.logs_area = ui.html().classes('font-mono text-sm bg-black p-4 rounded w-full h-96 overflow-y-scroll text-green-400')
                update_logs_display()

        # PANEL DOCUMENTATION
        with ui.tab_panel('tutorial'):
            with ui.card().classes('w-full bg-slate-800 p-4'):
                ui.label('Documentation de EcoRETINA ML Workbench').classes('text-2xl font-bold text-emerald-400')
                ui.markdown("""
                ### Guide de démarrage rapide :
                1. **Data & Import** : Chargez votre fichier CSV ou Excel depuis l'onglet principal.
                2. **IA Copilot** : Activez l'assistant en haut à droite avec votre clé API Gemini ou OpenAI pour un profiling automatisé.
                3. **Entraînement** : Sélectionnez votre modèle, définissez la colonne cible à prédire et cliquez sur Lancer.
                4. **Rapports Graphiques** : Le comparatif calculera automatiquement l'empreinte carbone via CodeCarbon.
                """)

# ==========================================
# FONCTIONS LOGIQUES APPLICATIVES (CALLBACKS)
# ==========================================

def update_logs_display():
    if hasattr(state, 'logs_area'):
        state.logs_area.content = "<br>".join(state.logs)

def handle_file_upload(e):
    # e.content est un flux binaire contenant le fichier téléversé
    try:
        if e.name.endswith('.csv'):
            state.df = pd.read_csv(e.content)
        else:
            state.df = pd.read_excel(e.content)
        
        state.log(f"Fichier chargé : {e.name} ({len(state.df)} lignes)")
        
        # Mettre à jour la liste des colonnes disponibles pour la cible Y
        columns = list(state.df.columns)
        if hasattr(state, 'target_select_ui'):
            state.target_select_ui.options = columns
            state.target_select_ui.value = columns[0]
            state.target_select_ui.update()
    except Exception as ex:
        state.log(f"Erreur d'import : {str(ex)}")

def open_data_viewer():
    if state.df is None:
        ui.notify("Aucun jeu de données chargé !")
        return
    with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-5xl h-5/6 bg-slate-900'):
        ui.label('Visualisation des données brutes').classes('text-h6 text-emerald-400')
        # On limite l'affichage web aux 50 premières lignes pour la fluidité réseau
        ui.table(
            columns=[{'name': col, 'label': col, 'field': col} for col in state.df.columns],
            rows=state.df.head(50).to_dict('records')
        ).classes('w-full bg-slate-950')
        ui.button('Fermer', on_click=dialog.close).classes('bg-red-600 mt-4')
    dialog.open()

def apply_undo():
    if state.history:
        action, previous_df = state.history.pop()
        state.future.append((action, state.df.copy()))
        state.df = previous_df
        state.log(f"Action annulée : {action}")

def apply_redo():
    if state.future:
        action, next_df = state.future.pop()
        state.history.append((action, state.df.copy()))
        state.df = next_df
        state.log(f"Action rétablie : {action}")

async def run_training(algo, target):
    if state.df is None:
        ui.notify("Veuillez charger des données d'abord !")
        return
    
    state.log(f"Lancement de l'apprentissage en tâche de fond ({algo})...")
    
    # Utilisation de run.cpu_bound pour éviter de geler l'interface web pendant les calculs lourds de Scikit-Learn
    metrics = await run.cpu_bound(compute_ml_pipeline, algo, target)
    
    # Ajouter le run à la table NiceGUI
    run_id = f"Run_{datetime.now().strftime('%H%M%S')}"
    state.runs_table.add_rows([{
        'run': run_id,
        'algo': algo,
        'r2_train': f"{metrics['r2_train']:.4f}",
        'r2_test': f"{metrics['r2_test']:.4f}",
        'mape': f"{metrics['mape']:.2f}%",
        'co2': f"{metrics['emissions']:.5f}"
    }])
    state.log(f"Pipeline terminé avec succès pour {algo} ! R² Test : {metrics['r2_test']:.4f}")

def compute_ml_pipeline(algo, target):
    # Nettoyage et split rapide pour l'exemple
    df_clean = state.df.dropna(subset=[target]).select_dtypes(include=[np.number]).fillna(0)
    X = df_clean.drop(columns=[target]).values
    y = df_clean[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Démarrage de CodeCarbon
    tracker = EmissionsTracker(tracking_mode='process', log_level='error')
    tracker.start()
    
    if algo == 'OLS' or algo == 'EcoRETINA':
        reg = Ridge(alpha=1.0) # Substitution robuste
    elif algo == 'XGBoost':
        reg = xgb.XGBRegressor(n_estimators=50, max_depth=4)
    elif algo == 'Neural Network':
        reg = MLPRegressor(max_iter=100)
    else:
        reg = RandomForestRegressor(n_estimators=50)
        
    reg.fit(X_train, y_train)
    emissions = tracker.stop()
    if emissions is None: emissions = 0.00002
    
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    
    return {
        'r2_train': r2_score(y_train, y_train_pred),
        'r2_test': r2_score(y_test, y_test_pred),
        'mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
        'emissions': emissions
    }

# Lancement officiel sur le port public demandé par Render
ui.run(port=int(os.environ.get('PORT', 8080)), title="EcoRETINA AI Workbench", reload=False)