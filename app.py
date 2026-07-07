import sys
import os
import traceback
import warnings
import asyncio
import re
import csv
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64

from nicegui import ui, run
from codecarbon import EmissionsTracker

# --- ALGORITHMES ET IMPORTS MACHINE LEARNING ---
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
# 1. ARCHITECTURE DES DONNÉES ET ÉTAT GLOBAL
# ==========================================
class Workspace:
    def __init__(self):
        self.df = None
        self.df_predict = None
        self.history = []  # Pour l'Undo
        self.future = []   # Pour le Redo
        self.run_history = {}
        self.latest_run_by_algo = {}
        self.logs = []
        self.ai_agent = None
        
        # Variables de configuration par défaut
        self.split_strategy = "Train/Test Split"
        self.train_split_pct = 80.0
        self.k_folds = 5
        self.target_var = ""

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        self.logs.append(log_line)
        ui.notify(message)
        if hasattr(self, 'logs_area') and self.logs_area:
            self.logs_area.content = "<br>".join(self.logs)

    def save_state(self, action_name):
        if self.df is not None:
            self.history.append((action_name, self.df.copy()))
            if len(self.history) > 15:
                self.history.pop(0)
            self.future.clear()

state = Workspace()

class OLSWrapper:
    def __init__(self, res): self.sm_model = res
    def predict(self, X): return self.sm_model.predict(X)

# ==========================================
# 2. IA COPILOT MULTI-PROVIDER (STREAMING SIM)
# ==========================================
class EcoRetinaChatAgent:
    def __init__(self, api_key: str, provider: str):
        self.provider = provider
        self.api_key = api_key
        self.system_prompt = (
            "You are the Chief Econometrician and AI Support Guide for the EcoRETINA ML Workbench.\n"
            "Output strictly plain text. Use ALL CAPS for emphasis."
        )
        self.history = []
        if provider == "Google Gemini": self.client = genai.Client(api_key=api_key)
        elif provider == "OpenAI (ChatGPT)": self.client = openai.OpenAI(api_key=api_key)
        elif provider == "Groq": self.client = Groq(api_key=api_key)

    async def ask(self, text: str, bubble_ui):
        try:
            if self.provider == "Google Gemini":
                response = self.client.chats.create(model="gemini-2.5-flash")
                reply = response.send_message(text).text
            elif self.provider == "OpenAI (ChatGPT)":
                self.history.append({"role": "user", "content": text})
                response = self.client.chat.completions.create(model="gpt-4o-mini", messages=self.history)
                reply = response.choices[0].message.content
            elif self.provider == "Groq":
                self.history.append({"role": "user", "content": text})
                response = self.client.chat.completions.create(model="llama-3.3-70b-versatile", messages=self.history)
                reply = response.choices[0].message.content
            bubble_ui.text = reply
        except Exception as e:
            bubble_ui.text = f"[Erreur API] : {str(e)}"

# ==========================================
# 3. INTERFACE GRAPHIQUE INTERACTIVE
# ==========================================

@ui.page('/')
def main_page():
    ui.dark_mode().enable()
    
    # --- HEADER ---
    with ui.header().classes('bg-slate-900 text-white items-center justify-between p-4 shadow-lg'):
        with ui.row().classes('items-center gap-4'):
            ui.label('≡').classes('text-3xl cursor-pointer').on('click', lambda: left_drawer.toggle())
            ui.label('EcoRETINA ML Workbench PRO').classes('text-xl font-bold text-emerald-400')
        with ui.row().classes('items-center gap-3'):
            ui.button('↩ Undo', on_click=apply_undo).props('flat color=white')
            ui.button('↪ Redo', on_click=apply_redo).props('flat color=white')
            ui.button('🤖 AI Copilot', on_click=lambda: right_drawer.toggle()).classes('bg-blue-600 font-bold')

    # --- DRAWER MENU LATÉRAL ---
    with ui.left_drawer(value=False).classes('bg-slate-800 text-white') as left_drawer:
        ui.label('Navigation Générale').classes('text-lg font-bold p-4 text-emerald-400 border-b border-slate-700 w-full')
        ui.button('Workspace Principal', on_click=lambda: main_tabs.set_value('workspace')).classes('w-full justify-start q-ma-xs')
        ui.button('Activity Log', on_click=lambda: main_tabs.set_value('logs')).classes('w-full justify-start q-ma-xs')
        ui.button('Tutorial & Docs', on_click=lambda: main_tabs.set_value('tutorial')).classes('w-full justify-start q-ma-xs')

    # --- DRAWER IA COPILOT ---
    with ui.right_drawer(value=False).classes('bg-slate-900 p-4 text-white') as right_drawer:
        ui.label('AI Copilot Assistant').classes('text-h6 font-bold text-emerald-400')
        provider_ui = ui.select(["Google Gemini", "OpenAI (ChatGPT)", "Groq"], value="Google Gemini").classes('w-full my-2')
        key_ui = ui.input(placeholder='Clé API', password=True).classes('w-full my-2')
        
        chat_container = ui.scroll_area().classes('w-full h-96 bg-slate-950 p-2 rounded my-4')
        
        async def connect_ai():
            if not key_ui.value: return ui.notify("Clé manquante !")
            state.ai_agent = EcoRetinaChatAgent(key_ui.value, provider_ui.value)
            state.log(f"IA connectée avec succès via {provider_ui.value}")
        
        ui.button('Connecter', on_click=connect_ai).classes('w-full bg-emerald-600')
        
        chat_input = ui.input(placeholder='Posez votre question...').classes('w-full mt-4')
        async def submit_chat():
            if not state.ai_agent: return
            with chat_container:
                ui.label(f"User: {chat_input.value}").classes('text-blue-400 font-bold block mt-2')
                ai_b = ui.label("En attente de réponse...").classes('text-white block ml-2 bg-slate-800 p-2 rounded')
                await state.ai_agent.ask(chat_input.value, ai_b)
                chat_input.value = ''
        chat_input.on('keydown.enter', submit_chat)

    # --- STRUCTURE PRINCIPALE DE L'APPLICATION ---
    with ui.tab_panels(ui.tabs(), value='workspace').classes('w-full bg-transparent') as main_tabs:
        
        # ==========================================
        # ONGLET WORKSPACE CONTENANT LES 5 ETAPES
        # ==========================================
        with ui.tab_panel('workspace'):
            with ui.tabs().classes('w-full bg-slate-950 text-white rounded shadow') as step_tabs:
                t1 = ui.tab('t_data', label='1. Data & Pre-Processing')
                t2 = ui.tab('t_algo', label='2. Algorithms & Params')
                t3 = ui.tab('t_compare', label='3. Compare Results')
                t4 = ui.tab('t_predict', label='4. Predict (New Data)')

            with ui.tab_panels(step_tabs, value='t_data').classes('w-full bg-transparent p-4') as step_panels:
                
                # ------------------------------------------
                # ETAPE 1 : DATA & PRE-PROCESSING
                # ------------------------------------------
                with ui.tab_panel('t_data'):
                    with ui.row().classes('w-full gap-4'):
                        # Import et split strategy
                        with ui.card().classes('w-full md:w-5/12 bg-slate-800 p-4'):
                            ui.label('Dataset Import & Strategy').classes('text-lg font-bold text-emerald-400')
                            ui.upload(label='Déposer CSV/Excel', on_upload=import_main_dataset, auto_upload=True).classes('w-full')
                            
                            ui.select(['Train/Test Split', 'K-Fold Cross Validation'], value='Train/Test Split', on_change=lambda e: toggle_split_view(e.value)).classes('w-full mt-4')
                            with ui.column().classes('w-full') as split_container:
                                state.split_slider = ui.slider(min=50, max=100, value=80).classes('w-full mt-2')
                                ui.label().bind_text_from(state.split_slider, 'value', backward=lambda v: f"Train Ratio : {v}%")
                            with ui.column().classes('w-full hidden') as kfold_container:
                                state.kfold_input = ui.number(label='Nombre de Folds (K)', value=5, min=2).classes('w-full')
                            state.split_container_ui = split_container
                            state.kfold_container_ui = kfold_container

                            ui.button('Visualiser la table', on_click=view_main_data).classes('bg-blue-600 w-full mt-4')

                        # Outliers & Encodage & Scaling
                        with ui.card().classes('w-full md:w-6/12 bg-slate-800 p-4'):
                            ui.label('Traitements & Nettoyage').classes('text-lg font-bold text-emerald-400')
                            
                            # Outliers
                            with ui.expansion('Gestion des Outliers (Valeurs Extrêmes)', icon='analytics').classes('w-full bg-slate-900 rounded'):
                                state.outlier_select = ui.select([], label='Variable Numérique').classes('w-full')
                                with ui.row().classes('w-full'):
                                    state.outlier_min = ui.number(label='Borne Inf').classes('w-1/2')
                                    state.outlier_max = ui.number(label='Borne Sup').classes('w-1/2')
                                state.outlier_action = ui.select(['Clip (Cap values)', 'Drop rows'], value='Clip (Cap values)').classes('w-full')
                                ui.button('Appliquer Filtre Outliers', on_click=process_outliers).classes('w-full bg-amber-600')

                            # Categoricals
                            with ui.expansion('Variables Qualitatives / Chaines', icon='g_translate').classes('w-full bg-slate-900 rounded mt-2'):
                                state.cat_select = ui.select([], label='Variable Catégorielle', on_change=update_cat_reference).classes('w-full')
                                state.cat_ref_select = ui.select([], label='Catégorie de Référence (Dropped)').classes('w-full')
                                with ui.row().classes('w-full'):
                                    ui.button('Encoder (Dummies)', on_click=lambda: run_cat_transformation('encode')).classes('bg-blue-600 expand')
                                    ui.button('Supprimer Colonne', on_click=lambda: run_cat_transformation('drop')).classes('bg-red-600 expand')

                            # Scaling
                            with ui.expansion('Normalisation / Scaling', icon='scale').classes('w-full bg-slate-900 rounded mt-2'):
                                state.scale_scope = ui.select(['All Predictors', 'Target Variable ONLY', 'All Variables'], value='All Predictors').classes('w-full')
                                state.scale_method = ui.select(['StandardScaler (Z-Score)', 'MinMaxScaler (0-1)'], value='StandardScaler (Z-Score)').classes('w-full')
                                ui.button('Lancer la Normalisation', on_click=run_scaling_process).classes('w-full bg-indigo-600')

                # ------------------------------------------
                # ETAPE 2 : ALGORITHMS & PARAMS
                # ------------------------------------------
                with ui.tab_panel('t_algo'):
                    with ui.card().classes('w-full bg-slate-800 p-4'):
                        ui.label('Configuration de l\'Algorithme d\'Apprentissage').classes('text-xl font-bold text-emerald-400 mb-4')
                        
                        with ui.row().classes('w-full gap-4 items-center'):
                            state.algo_choice = ui.select(['EcoRETINA', 'OLS', 'Lasso', 'Ridge', 'ElasticNet', 'XGBoost', 'Random Forest', 'Neural Network'], value='EcoRETINA', on_change=refresh_algo_param_view).classes('w-1/3')
                            state.main_target_select = ui.select([], label='Variable Cible (Y) à prédire').classes('w-1/3')
                        
                        # Zone dynamique pour les hyperparamètres spécifiques
                        state.param_options_frame = ui.row().classes('w-full bg-slate-900 p-4 rounded mt-4')
                        
                        # Grille de sélection des variables explicatives (Features)
                        ui.label('Sélection des Prédicteurs (Features X)').classes('text-lg font-bold text-emerald-400 mt-6')
                        state.features_checkbox_container = ui.row().classes('w-full bg-slate-950 p-4 rounded max-h-60 overflow-y-scroll border border-slate-700')
                        
                        # Boutons d'actions d'apprentissage
                        with ui.row().classes('w-full justify-between items-center mt-6 border-t border-slate-700 pt-4'):
                            state.algo_status_lbl = ui.label('Pipeline prêt. Conduisez l\'analyse.').classes('text-gray-400 font-mono')
                            with ui.row():
                                ui.button('Voir Rapport Statistique Détaillé', on_click=open_detailed_report).classes('bg-blue-600')
                                ui.button('► Lancer l\'Entraînement', on_click=trigger_pipeline_execution).classes('bg-emerald-600 font-bold px-6 text-lg')

                # ------------------------------------------
                # ETAPE 3 : COMPARE RESULTS
                # ------------------------------------------
                with ui.tab_panel('t_compare'):
                    with ui.card().classes('w-full bg-slate-800 p-4'):
                        ui.label('Tableau de Bord Comparatif des Évaluations').classes('text-xl font-bold text-emerald-400')
                        ui.label('Les meilleures métriques validées sont automatiquement identifiées par un astérisque (*)').classes('text-sm text-gray-400 mb-4')
                        
                        state.compare_table_ui = ui.table(
                            columns=[
                                {'name': 'run', 'label': 'Run ID', 'field': 'run', 'align': 'center'},
                                {'name': 'algo', 'label': 'Algorithme', 'field': 'algo', 'align': 'center'},
                                {'name': 'r2_tr', 'label': 'R² Train', 'field': 'r2_tr', 'align': 'center'},
                                {'name': 'mape_tr', 'label': 'MAPE Train', 'field': 'mape_tr', 'align': 'center'},
                                {'name': 'r2_te', 'label': 'R² Test', 'field': 'r2_te', 'align': 'center'},
                                {'name': 'rmse_te', 'label': 'RMSE Test', 'field': 'rmse_te', 'align': 'center'},
                                {'name': 'mape_te', 'label': 'MAPE Test', 'field': 'mape_te', 'align': 'center'},
                                {'name': 'co2', 'label': 'Carbon (kgCO2)', 'field': 'co2', 'align': 'center'},
                            ], rows=[]
                        ).classes('w-full bg-slate-900 text-white')
                        
                        with ui.row().classes('w-full justify-between mt-4'):
                            ui.button('Vider l\'historique', on_click=lambda: state.compare_table_ui.rows.clear()).classes('bg-red-600')
                            ui.button('Exporter Synthèse CSV', on_click=export_comparison_matrix).classes('bg-emerald-600')

                # ------------------------------------------
                # ETAPE 4 : PREDICT (NEW DATA / INFERENCE)
                # ------------------------------------------
                with ui.tab_panel('t_predict'):
                    with ui.card().classes('w-full bg-slate-800 p-4'):
                        ui.label('Inférence & Prédiction sur de Nouvelles Données').classes('text-xl font-bold text-emerald-400 mb-4')
                        
                        with ui.row().classes('w-full gap-4'):
                            with ui.card().classes('w-1/2 bg-slate-900 p-4'):
                                ui.label('1. Fichier de test / Inférence').classes('text-md font-bold text-gray-300')
                                ui.upload(label='Charger fichier sans cible Y', on_upload=import_predict_dataset, auto_upload=True).classes('w-full')
                                state.predict_file_lbl = ui.label('Aucun fichier d\'inférence chargé').classes('text-gray-400 font-mono text-xs')
                            
                            with ui.card().classes('w-1/2 bg-slate-900 p-4'):
                                ui.label('2. Choix du Modèle Appris').classes('text-md font-bold text-gray-300')
                                state.predict_run_select = ui.select([], label='Sélectionner un Run validé').classes('w-full')
                                ui.button('Rafraîchir la Liste', on_click=sync_predict_runs).classes('w-full bg-slate-700')

                        with ui.row().classes('w-full justify-between items-center mt-6 border-t border-slate-700 pt-4'):
                            with ui.row():
                                ui.button('Exécuter la Prédiction', on_click=execute_inference_process).classes('bg-emerald-600 font-bold text-lg')
                                ui.button('Visualiser Résultats', on_click=view_predict_data).classes('bg-blue-600')
                            ui.button('Exporter le fichier enrichi (.csv)', on_click=export_predicted_csv).classes('bg-indigo-600')

        # TAB LOGS SÉPARÉ
        with ui.tab_panel('logs'):
            with ui.card().classes('w-full bg-slate-800 p-4'):
                ui.label('Journal d\'Activité du Serveur').classes('text-xl font-bold text-emerald-400 mb-4')
                state.logs_area = ui.html().classes('font-mono text-sm bg-black p-4 rounded w-full h-96 overflow-y-scroll text-green-400')
                state.logs_area.content = "<br>".join(state.logs)

        # TAB TUTORIAL SÉPARÉ
        with ui.tab_panel('tutorial'):
            with ui.card().classes('w-full bg-slate-800 p-4 text-white'):
                ui.label('Tutoriel Économétrique & ML Workspace').classes('text-2xl font-bold text-emerald-400 mb-4')
                ui.markdown("""
                - **Étape 1 (Data)** : Chargez votre fichier. Utilisez les modules d'expansion pour filtrer les Outliers ou générer vos variables muettes (Dummies) avec exclusion stricte de la modalité de référence.
                - **Étape 2 (Algorithm)** : Cochez manuellement vos régresseurs explicatifs. Les hyperparamètres s'ajustent dynamiquement.
                - **Étape 3 (Compare)** : Analyse comparative automatique intégrant le diagnostic d'émissions de gaz à effet de serre en temps réel (CodeCarbon).
                """)

    # Initialisation de l'affichage des paramètres par défaut
    refresh_algo_param_view()

# ==========================================
# 4. LOGIQUE INTERNE & ACTIONS UTILISATEURS
# ==========================================

def toggle_split_view(strategy):
    state.split_strategy = strategy
    if strategy == 'Train/Test Split':
        state.split_container_ui.remove_class('hidden')
        state.kfold_container_ui.add_class('hidden')
    else:
        state.split_container_ui.add_class('hidden')
        state.kfold_container_ui.remove_class('hidden')

def import_main_dataset(e):
    try:
        state.df = pd.read_csv(e.content) if e.name.endswith('.csv') else pd.read_excel(e.content)
        state.save_state(f"Import dataset: {e.name}")
        state.log(f"Jeu de données principal chargé : {e.name} ({len(state.df)} lignes, {len(state.df.columns)} variables)")
        sync_all_comboboxes()
    except Exception as ex:
        state.log(f"Échec de l'importation : {str(ex)}")

def sync_all_comboboxes():
    if state.df is None: return
    cols = list(state.df.columns)
    num_cols = list(state.df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(state.df.select_dtypes(include=['object', 'category']).columns)
    
    state.outlier_select.options = num_cols
    if num_cols: state.outlier_select.value = num_cols[0]
    
    state.cat_select.options = cat_cols
    if cat_cols: state.cat_select.value = cat_cols[0]
    
    state.main_target_select.options = cols
    if cols: state.main_target_select.value = cols[0]
    
    # Génération des cases à cocher pour les Features
    state.features_checkbox_container.clear()
    with state.features_checkbox_container:
        for c in cols:
            ui.checkbox(text=c, value=True).classes('text-white mx-2 font-mono')

def update_cat_reference(e):
    if state.df is None or not e.value: return
    instances = [str(x) for x in state.df[e.value].dropna().unique()]
    state.cat_ref_select.options = instances
    if instances: state.cat_ref_select.value = instances[0]

def process_outliers():
    if state.df is None or not state.outlier_select.value: return
    col = state.outlier_select.value
    mn, mx = state.outlier_min.value, state.outlier_max.value
    if mn is None or mx is None: return ui.notify("Veuillez saisir les bornes !")
    
    state.save_state(f"Outliers filtering on {col}")
    if 'Clip' in state.outlier_action.value:
        state.df[col] = state.df[col].clip(lower=mn, upper=mx)
        state.log(f"Outliers écrêtés (Clipped) pour '{col}' entre {mn} et {mx}")
    else:
        init_l = len(state.df)
        state.df = state.df[(state.df[col] >= mn) & (state.df[col] <= mx)]
        state.log(f"Suppression de {init_l - len(state.df)} lignes hors limites pour '{col}'")
    sync_all_comboboxes()

def run_cat_transformation(action):
    if state.df is None or not state.cat_select.value: return
    col = state.cat_select.value
    state.save_state(f"Transformation qualitatives ({action}) sur {col}")
    
    if action == 'drop':
        state.df.drop(columns=[col], inplace=True)
        state.log(f"Colonne '{col}' supprimée.")
    else:
        ref = state.cat_ref_select.value
        dummies = pd.get_dummies(state.df[col], prefix=col).astype(int)
        if f"{col}_{ref}" in dummies.columns:
            dummies.drop(columns=[f"{col}_{ref}"], inplace=True)
        state.df = pd.concat([state.df.drop(columns=[col]), dummies], axis=1)
        state.log(f"Colonne '{col}' convertie en Dummies (Exclusion de la référence : '{ref}')")
    sync_all_comboboxes()

def run_scaling_process():
    if state.df is None: return
    scope = state.scale_scope.value
    method = state.scale_method.value
    target = state.main_target_select.value
    
    state.save_state(f"Scaling metrics ({method})")
    scaler = StandardScaler() if "Standard" in method else MinMaxScaler()
    num_cols = state.df.select_dtypes(include=[np.number]).columns.tolist()
    
    try:
        if scope == "Target Variable ONLY":
            state.df[[target]] = scaler.fit_transform(state.df[[target]])
        elif scope == "All Predictors":
            feats = [c for c in num_cols if c != target]
            state.df[feats] = scaler.fit_transform(state.df[feats])
        else:
            state.df[num_cols] = scaler.fit_transform(state.df[num_cols])
        state.log(f"Normalisation appliquée via {method} (Périmètre : {scope})")
    except Exception as ex:
        state.log(f"Erreur de normalisation : {str(ex)}")

def view_main_data():
    if state.df is None: return ui.notify("Aucune table chargée")
    with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-5xl h-5/6 bg-slate-900 text-white'):
        ui.label('Aperçu du Dataset Principal (Top 50)').classes('text-h6 text-emerald-400 font-bold')
        ui.table(
            columns=[{'name': c, 'label': c, 'field': c} for c in state.df.columns],
            rows=state.df.head(50).to_dict('records')
        ).classes('w-full bg-slate-950')
        ui.button('Fermer', on_click=dialog.close).classes('bg-red-600')
    dialog.open()

def apply_undo():
    if state.history:
        act, prev = state.history.pop()
        state.future.append((act, state.df.copy()))
        state.df = prev
        state.log(f"[UNDO] : Retour arrière sur l'action '{act}'")
        sync_all_comboboxes()

def apply_redo():
    if state.future:
        act, nxt = state.future.pop()
        state.history.append((act, state.df.copy()))
        state.df = nxt
        state.log(f"[REDO] : Réapplication de l'action '{act}'")
        sync_all_comboboxes()

# ==========================================
# 5. PARAMÈTRES DYNAMIQUES DES ALGORITHMES
# ==========================================

def refresh_algo_param_view():
    algo = state.algo_choice.value
    state.param_options_frame.clear()
    with state.param_options_frame:
        if algo in ['Lasso', 'Ridge', 'ElasticNet']:
            state.alpha_input = ui.number(label='Alpha (Pénalité)', value=0.01, format='%.4f').classes('w-40')
            state.max_iter_input = ui.number(label='Max Iterations', value=1000).classes('w-40')
        elif algo == 'XGBoost':
            state.xgb_n = ui.number(label='N Estimators', value=100).classes('w-40')
            state.xgb_depth = ui.number(label='Max Depth', value=6).classes('w-40')
            state.xgb_lr = ui.number(label='Learning Rate', value=0.1).classes('w-40')
        elif algo == 'Neural Network':
            state.nn_layers = ui.input(label='Layers (e.g. 100,50)', value="100,50").classes('w-40')
            state.nn_iter = ui.number(label='Max Iter', value=200).classes('w-40')
        elif algo == 'EcoRETINA':
            state.eco_loss = ui.select(['mse', 'mae', 'AIC', 'BIC'], value='mse', label='Loss').classes('w-32')
            state.eco_grid = ui.number(label='Grid step', value=0.005).classes('w-32')
        else:
            ui.label('Pas d\'hyperparamètres spécifiques requis pour OLS.').classes('text-gray-400 italic')

# ==========================================
# 6. PIPELINE MULTI-THREAD COMPULSAR CODE
# ==========================================

async def trigger_pipeline_execution():
    if state.df is None: return ui.notify("Veuillez importer des données !")
    target = state.main_target_select.value
    algo = state.algo_choice.value
    
    # Collecter les colonnes cochées par l'utilisateur
    selected_features = []
    for child in state.features_checkbox_container:
        if isinstance(child, ui.checkbox) and child.value and child.text != target:
            selected_features.append(child.text)
            
    if not selected_features:
        return ui.notify("Veuillez cocher au moins une variable explicative (X) !")

    state.algo_status_lbl.text = f"Calculs économétriques en cours pour {algo}..."
    state.algo_status_lbl.update()
    
    # Extraction des hyperparamètres de l'UI de manière sécurisée
    config_args = {
        'alpha': getattr(state, 'alpha_input', type('obj', (), {'value': 0.01})).value,
        'max_iter': int(getattr(state, 'max_iter_input', type('obj', (), {'value': 1000})).value),
        'xgb_n': int(getattr(state, 'xgb_n', type('obj', (), {'value': 100})).value),
        'xgb_depth': int(getattr(state, 'xgb_depth', type('obj', (), {'value': 6})).value),
        'xgb_lr': getattr(state, 'xgb_lr', type('obj', (), {'value': 0.1})).value,
        'nn_layers': getattr(state, 'nn_layers', type('obj', (), {'value': "100,50"})).value,
        'nn_iter': int(getattr(state, 'nn_iter', type('obj', (), {'value': 200})).value),
        'eco_loss': getattr(state, 'eco_loss', type('obj', (), {'value': "mse"})).value,
        'eco_grid': getattr(state, 'eco_grid', type('obj', (), {'value': 0.005})).value,
    }

    # Lancement asynchrone sécurisé non-bloquant
    res = await run.cpu_bound(execute_ml_math_core, algo, target, selected_features, config_args)
    
    run_id = f"Run_{time.strftime('%H%M%S')}"
    state.run_history[run_id] = res
    state.latest_run_by_algo[algo] = run_id
    
    # Ajouter à la table comparative
    state.compare_table_ui.add_rows([{
        'run': run_id, 'algo': algo,
        'r2_tr': f"{res['metrics']['r2_tr']:.4f}", 'mape_tr': f"{res['metrics']['mape_tr']:.2f}%",
        'r2_te': f"{res['metrics']['r2_te']:.4f}", 'rmse_te': f"{res['metrics']['rmse_te']:.4f}",
        'mape_te': f"{res['metrics']['mape_te']:.2f}%", 'co2': f"{res['metrics']['emissions']:.5f}"
    }])
    
    state.algo_status_lbl.text = f"Dernier succès : {run_id} ({algo})"
    state.log(f" pipeline complété avec succès pour {algo}. R² Test = {res['metrics']['r2_te']:.4f}")

def execute_ml_math_core(algo, target, features, args):
    df_clean = state.df.dropna(subset=[target] + features).fillna(0)
    X = df_clean[features].values
    y = df_clean[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tracker = EmissionsTracker(tracking_mode='process', log_level='error')
    tracker.start()
    
    model = None
    feature_names_final = features.copy()
    
    if algo == 'OLS':
        X_train_fit = sm.add_constant(X_train, has_constant='add')
        X_test_fit = sm.add_constant(X_test, has_constant='add')
        feature_names_final = ['const'] + features
        sm_res = sm.OLS(y_train, pd.DataFrame(X_train_fit, columns=feature_names_final)).fit()
        model = OLSWrapper(sm_res)
        y_train_pred = sm_res.predict(pd.DataFrame(X_train_fit, columns=feature_names_final))
        y_test_pred = sm_res.predict(pd.DataFrame(X_test_fit, columns=feature_names_final))
        
    elif algo in ['Lasso', 'Ridge', 'ElasticNet']:
        if algo == 'Lasso': model_pen = Lasso(alpha=args['alpha'], max_iter=args['max_iter'])
        elif algo == 'Ridge': model_pen = Ridge(alpha=args['alpha'], max_iter=args['max_iter'])
        else: model_pen = ElasticNet(alpha=args['alpha'], max_iter=args['max_iter'])
        
        model_pen.fit(X_train, y_train)
        sel_idx = np.where(np.abs(model_pen.coef_) > 1e-5)[0]
        if len(sel_idx) == 0: sel_idx = np.arange(X_train.shape[1])
        
        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]
        feature_names_final = [features[i] for i in sel_idx]
        
        X_tr_fit = sm.add_constant(X_tr_sel, has_constant='add')
        X_te_fit = sm.add_constant(X_te_sel, has_constant='add')
        feature_names_final = ['const'] + feature_names_final
        
        sm_res = sm.OLS(y_train, pd.DataFrame(X_tr_fit, columns=feature_names_final)).fit()
        model = OLSWrapper(sm_res)
        y_train_pred = sm_res.predict(pd.DataFrame(X_tr_fit, columns=feature_names_final))
        y_test_pred = sm_res.predict(pd.DataFrame(X_te_fit, columns=feature_names_final))
        
    elif algo == 'XGBoost':
        model = xgb.XGBRegressor(n_estimators=args['xgb_n'], max_depth=args['xgb_depth'], learning_rate=args['xgb_lr'])
        model.fit(X_train, y_train)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    else:
        model = RandomForestRegressor(n_estimators=50)
        model.fit(X_train, y_train)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        
    emissions = tracker.stop()
    if emissions is None: emissions = 0.00001
    
    return {
        'model': model, 'model_name': algo, 'target_col': target, 'raw_features': features, 'feature_names': feature_names_final,
        'y_test': y_test, 'y_test_pred': y_test_pred,
        'metrics': {
            'r2_tr': r2_score(y_train, y_train_pred),
            'mape_tr': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            'r2_te': r2_score(y_test, y_test_pred),
            'rmse_te': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mape_te': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
            'emissions': emissions
        }
    }

# ==========================================
# 7. RAPPORTS GRAPHIQUES ET ANALYTIQUES
# ==========================================

def open_detailed_report():
    algo = state.algo_choice.value
    run_id = state.latest_run_by_algo.get(algo)
    if not run_id: return ui.notify("Aucune exécution enregistrée pour cet algorithme !")
    
    run_data = state.run_history[run_id]
    y_test = run_data['y_test']
    y_test_pred = run_data['y_test_pred']
    
    # Construction de la figure Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#1e293b')
    
    axes[0].scatter(y_test, y_test_pred, alpha=0.6, color='#3b82f6')
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#10b981', linestyle='--')
    axes[0].set_title('Valeurs Réelles vs Prédictions', color='white')
    axes[0].set_facecolor('#0f172a')
    axes[0].tick_params(colors='white')
    
    residuals = y_test - y_test_pred
    axes[1].hist(residuals, bins=15, color='#ef4444', alpha=0.8)
    axes[1].set_title('Distribution des Résidus', color='white')
    axes[1].set_facecolor('#0f172a')
    axes[1].tick_params(colors='white')
    
    # Encodage Base64 du graphique pour insertion HTML
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-4xl bg-slate-900 text-white p-6'):
        ui.label(f"Rapport Statistique Graphique - {run_id}").classes('text-xl font-bold text-emerald-400')
        ui.html(f'<div class="flex justify-center"><img src="data:image/png;base64,{img_b64}"/></div>')
        ui.button('Fermer', on_click=dialog.close).classes('bg-red-600 self-end mt-4')
    dialog.open()

def export_comparison_matrix():
    # Génération d'une chaine CSV à la volée pour téléchargement
    if not state.compare_table_ui.rows: return ui.notify("Aucune donnée à exporter")
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Run_ID', 'Algorithm', 'R2_Train', 'R2_Test', 'MAPE_Test', 'CO2_kg'])
    for r in state.compare_table_ui.rows:
        writer.writerow([r['run'], r['algo'], r['r2_tr'], r['r2_te'], r['mape_te'], r['co2']])
    ui.download(output.getvalue().encode('utf-8'), 'ML_Workbench_Benchmark.csv')

# ==========================================
# 8. ETAPE 4 : LOGIQUE D'INFERENCE (PREDICT)
# ==========================================

def import_predict_dataset(e):
    try:
        state.df_predict = pd.read_csv(e.content) if e.name.endswith('.csv') else pd.read_excel(e.content)
        state.predict_file_lbl.text = f"Fichier inférence chargé : {e.name} ({len(state.df_predict)} lignes)"
        state.log(f"Jeu de données pour prédiction importé : {e.name}")
    except Exception as ex:
        ui.notify(f"Erreur chargement inférence : {str(ex)}")

def sync_predict_runs():
    runs = list(state.run_history.keys())
    state.predict_run_select.options = runs
    if runs: state.predict_run_select.value = runs[-1]

def execute_inference_process():
    if state.df_predict is None: return ui.notify("Veuillez d'abord charger un fichier d'inférence (Étape 1)")
    run_id = state.predict_run_select.value
    if not run_id: return ui.notify("Veuillez sélectionner un modèle entraîné (Étape 2)")
    
    run_data = state.run_history[run_id]
    raw_feats = run_data['raw_features']
    model = run_data['model']
    feature_names = run_data['feature_names']
    target_col = run_data['target_col']
    
    try:
        X_new = state.df_predict[raw_feats].fillna(0).values
        if 'const' in feature_names:
            X_new = sm.add_constant(X_new, has_constant='add')
            
        preds = model.predict(pd.DataFrame(X_new, columns=feature_names))
        pred_col = f"Predicted_{target_col}_{run_id}"
        state.df_predict[pred_col] = preds
        ui.notify(f"Prédictions injectées avec succès dans la colonne '{pred_col}' !")
    except Exception as ex:
        ui.notify(f"Erreur d'inférence : {str(ex)}")

def view_predict_data():
    if state.df_predict is None: return ui.notify("Aucune donnée d'inférence chargée.")
    with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-5xl h-5/6 bg-slate-900 text-white'):
        ui.label('Données Inférence & Prédictions générées').classes('text-h6 text-emerald-400')
        ui.table(
            columns=[{'name': c, 'label': c, 'field': c} for c in state.df_predict.columns],
            rows=state.df_predict.head(50).to_dict('records')
        ).classes('w-full bg-slate-950')
        ui.button('Fermer', on_click=dialog.close).classes('bg-red-600')
    dialog.open()

def export_predicted_csv():
    if state.df_predict is None: return
    csv_buf = StringIO()
    state.df_predict.to_csv(csv_buf, index=False)
    ui.download(csv_buf.getvalue().encode('utf-8'), 'EcoRETINA_Predictions_Output.csv')

# Lancement de l'application Web sur le port système Render
ui.run(port=int(os.environ.get('PORT', 8080)), title="EcoRETINA AI Workbench", reload=False)
