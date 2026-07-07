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
        self.history = []  
        self.future = []   
        self.run_history = {}
        self.latest_run_by_algo = {}
        self.logs = []
        self.ai_agent = None
        
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
# 2. IA COPILOT MULTI-PROVIDER
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
# 3. INTERFACE DESIGN INCURVÉ & MODERNE
# ==========================================

@ui.page('/')
def main_page():
    ui.dark_mode().enable()
    
    ui.add_head_html('''
    <style>
        .q-btn { border-radius: 12px !important; text-transform: none !important; font-weight: 600 !important; }
        .q-field { border-radius: 14px !important; }
        .q-field__control { border-radius: 14px !important; }
        .q-tab { border-radius: 12px 12px 0 0 !important; margin: 0 4px; }
        .q-panel { border-radius: 20px !important; }
    </style>
    ''')
    
    # --- HEADER ARRONDI ---
    with ui.header().classes('bg-slate-900/80 backdrop-blur-md text-white items-center justify-between p-4 shadow-xl m-4 rounded-2xl border border-slate-800'):
        with ui.row().classes('items-center gap-4'):
            ui.label('≡').classes('text-3xl cursor-pointer hover:text-emerald-400 transition-colors').on('click', lambda: left_drawer.toggle())
            ui.label('EcoRETINA Intelligence').classes('text-xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-300')
        with ui.row().classes('items-center gap-3'):
            ui.button('↩ Undo', on_click=apply_undo).props('flat color=white').classes('hover:bg-slate-800 rounded-xl')
            ui.button('↪ Redo', on_click=apply_redo).props('flat color=white').classes('hover:bg-slate-800 rounded-xl')
            ui.button('🤖 AI Copilot', on_click=lambda: right_drawer.toggle()).classes('bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl font-bold shadow-lg shadow-blue-500/20')

    # --- DRAWER MENU LATÉRAL MODERNE ---
    with ui.left_drawer(value=False).classes('bg-slate-900/90 backdrop-blur-md p-4 text-white rounded-r-3xl border-r border-slate-800') as left_drawer:
        ui.label('Navigation').classes('text-sm uppercase tracking-wider font-bold p-2 text-slate-400 border-b border-slate-800 w-full mb-4')
        ui.button('Workspace Principal', on_click=lambda: main_tabs.set_value('workspace')).classes('w-full justify-start rounded-xl mb-2 py-3 bg-slate-800/50 hover:bg-slate-800')
        ui.button('Activity Log', on_click=lambda: main_tabs.set_value('logs')).classes('w-full justify-start rounded-xl mb-2 py-3 bg-slate-800/50 hover:bg-slate-800')
        ui.button('Tutorial & Docs', on_click=lambda: main_tabs.set_value('tutorial')).classes('w-full justify-start rounded-xl mb-2 py-3 bg-slate-800/50 hover:bg-slate-800')

    # --- DRAWER IA COPILOT ARRONDI ---
    with ui.right_drawer(value=False).classes('bg-slate-900/90 backdrop-blur-md p-4 text-white rounded-l-3xl border-l border-slate-800') as right_drawer:
        ui.label('AI Assistant').classes('text-lg font-black text-emerald-400 mb-2')
        provider_ui = ui.select(["Google Gemini", "OpenAI (ChatGPT)", "Groq"], value="Google Gemini").classes('w-full rounded-xl')
        key_ui = ui.input(placeholder='Clé API', password=True).classes('w-full rounded-xl')
        
        chat_container = ui.scroll_area().classes('w-full h-96 bg-slate-950/60 p-3 rounded-2xl border border-slate-800 my-4 shadow-inner')
        
        async def connect_ai():
            if not key_ui.value: return ui.notify("Clé manquante !")
            state.ai_agent = EcoRetinaChatAgent(key_ui.value, provider_ui.value)
            state.log(f"IA connectée avec succès via {provider_ui.value}")
        
        ui.button('Connecter l\'IA', on_click=connect_ai).classes('w-full bg-emerald-600 rounded-xl font-bold shadow-lg shadow-emerald-500/20')
        
        chat_input = ui.input(placeholder='Posez votre question...').classes('w-full mt-4 rounded-xl')
        async def submit_chat():
            if not chat_input.value: return
            if not state.ai_agent: return
            with chat_container:
                ui.label(f"User: {chat_input.value}").classes('text-blue-400 font-bold block mt-2 text-sm')
                ai_b = ui.label("Analyse en cours...").classes('text-slate-200 block ml-2 bg-slate-800/80 p-3 rounded-2xl text-sm border border-slate-700/50')
                await state.ai_agent.ask(chat_input.value, ai_b)
                chat_input.value = ''
        chat_input.on('keydown.enter', submit_chat)

    # --- BLOC DE CONTENU GENERAL ---
    with ui.tab_panels(ui.tabs(), value='workspace').classes('w-full bg-transparent px-4') as main_tabs:
        
        with ui.tab_panel('workspace'):
            with ui.tabs().classes('w-full bg-slate-900/40 p-2 rounded-2xl border border-slate-800/60 text-white') as step_tabs:
                t1 = ui.tab('t_data', label='1. Data & Pre-Processing').classes('rounded-xl')
                t2 = ui.tab('t_algo', label='2. Algorithms & Params').classes('rounded-xl')
                t3 = ui.tab('t_compare', label='3. Compare Results').classes('rounded-xl')
                t4 = ui.tab('t_predict', label='4. Predict (New Data)').classes('rounded-xl')

            with ui.tab_panels(step_tabs, value='t_data').classes('w-full bg-transparent pt-4 overflow-visible') as step_panels:
                
                # ------------------------------------------
                # ETAPE 1 : DATA & PRE-PROCESSING
                # ------------------------------------------
                with ui.tab_panel('t_data'):
                    with ui.row().classes('w-full gap-6'):
                        with ui.card().classes('w-full md:w-[48%] bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                            ui.label('Dataset Import & Sample Strategy').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-2')
                            
                            # METHODE UNIVERSELLE : On extrait directement les octets réseau
                            ui.upload(label='Glissez-déposez votre CSV', on_upload=lambda e: import_main_dataset_from_bytes(e.content.read()), auto_upload=True).classes('w-full rounded-2xl')
                            
                            ui.select(['Train/Test Split', 'K-Fold Cross Validation'], value='Train/Test Split', on_change=lambda e: toggle_split_view(e.value)).classes('w-full mt-4 rounded-xl')
                            with ui.column().classes('w-full') as split_container:
                                state.split_slider = ui.slider(min=50, max=100, value=80).classes('w-full mt-2')
                                ui.label().bind_text_from(state.split_slider, 'value', backward=lambda v: f"Ratio d'apprentissage : {v}%")
                            with ui.column().classes('w-full hidden') as kfold_container:
                                state.kfold_input = ui.number(label='Nombre de Folds (K)', value=5, min=2).classes('w-full rounded-xl')
                            state.split_container_ui = split_container
                            state.kfold_container_ui = kfold_container

                            ui.button('Visualiser la table de données', on_click=view_main_data).classes('bg-blue-600/90 w-full mt-6 rounded-xl py-2 font-bold')

                        with ui.card().classes('w-full md:w-[48%] bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                            ui.label('Traitements & Nettoyage Avancé').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-4')
                            
                            with ui.expansion('Gestion des Outliers (Valeurs Extrêmes)', icon='analytics').classes('w-full bg-slate-950/50 border border-slate-800 rounded-xl mb-3'):
                                state.outlier_select = ui.select([], label='Variable Numérique').classes('w-full')
                                with ui.row().classes('w-full gap-2'):
                                    state.outlier_min = ui.number(label='Borne Inf').classes('w-[47%]')
                                    state.outlier_max = ui.number(label='Borne Sup').classes('w-[47%]')
                                state.outlier_action = ui.select(['Clip (Cap values)', 'Drop rows'], value='Clip (Cap values)').classes('w-full')
                                ui.button('Appliquer le filtre Outliers', on_click=process_outliers).classes('w-full bg-amber-600 rounded-xl mt-2')

                            with ui.expansion('Variables Qualitatives / Encodage', icon='g_translate').classes('w-full bg-slate-950/50 border border-slate-800 rounded-xl mb-3'):
                                state.cat_select = ui.select([], label='Variable Catégorielle', on_change=update_cat_reference).classes('w-full')
                                state.cat_ref_select = ui.select([], label='Catégorie de Référence (Dropped)').classes('w-full')
                                with ui.row().classes('w-full gap-2 mt-2'):
                                    ui.button('Encoder en Dummies', on_click=lambda: run_cat_transformation('encode')).classes('bg-blue-600 w-[48%] rounded-xl')
                                    ui.button('Supprimer Colonne', on_click=lambda: run_cat_transformation('drop')).classes('bg-red-600/80 w-[48%] rounded-xl')

                            with ui.expansion('Normalisation / Scaling', icon='scale').classes('w-full bg-slate-950/50 border border-slate-800 rounded-xl'):
                                state.scale_scope = ui.select(['All Predictors', 'Target Variable ONLY', 'All Variables'], value='All Predictors').classes('w-full')
                                state.scale_method = ui.select(['StandardScaler (Z-Score)', 'MinMaxScaler (0-1)'], value='StandardScaler (Z-Score)').classes('w-full')
                                ui.button('Exécuter la mise à l\'échelle', on_click=run_scaling_process).classes('w-full bg-indigo-600 rounded-xl mt-2')

                # ------------------------------------------
                # ETAPE 2 : ALGORITHMS & PARAMS
                # ------------------------------------------
                with ui.tab_panel('t_algo'):
                    with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                        ui.label('Configuration de l\'Algorithme').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-4')
                        
                        with ui.row().classes('w-full gap-4 items-center'):
                            state.algo_choice = ui.select(['EcoRETINA', 'OLS', 'Lasso', 'Ridge', 'ElasticNet', 'XGBoost', 'Random Forest', 'Neural Network'], value='EcoRETINA', on_change=refresh_algo_param_view).classes('w-1/3 rounded-xl')
                            state.main_target_select = ui.select([], label='Variable Cible (Y) à modéliser').classes('w-1/3 rounded-xl')
                        
                        state.param_options_frame = ui.row().classes('w-full bg-slate-950/50 p-4 rounded-xl border border-slate-800 mt-4')
                        
                        ui.label('Sélection des Predictors (X)').classes('text-md uppercase tracking-wider font-bold text-slate-400 mt-6 mb-2')
                        state.features_checkbox_container = ui.row().classes('w-full bg-slate-950/80 p-4 rounded-2xl max-h-60 overflow-y-scroll border border-slate-800')
                        
                        with ui.row().classes('w-full justify-between items-center mt-6 border-t border-slate-800 pt-4'):
                            state.algo_status_lbl = ui.label('Pipeline prêt. Conduisez l\'analyse.').classes('text-slate-400 font-mono text-sm')
                            with ui.row().classes('gap-3'):
                                ui.button('Rapport Graphique Détaillé', on_click=open_detailed_report).classes('bg-blue-600/90 rounded-xl font-bold')
                                ui.button('► Lancer l\'Apprentissage', on_click=trigger_pipeline_execution).classes('bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl font-bold px-6 shadow-lg shadow-emerald-500/10')

                # ------------------------------------------
                # ETAPE 3 : COMPARE RESULTS
                # ------------------------------------------
                with ui.tab_panel('t_compare'):
                    with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                        ui.label('Benchmark Comparatif Global').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-1')
                        ui.label('Les optimaux statistiques sont marqués d\'une étoile (*)').classes('text-xs text-slate-400 mb-4')
                        
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
                        ).classes('w-full bg-slate-950 text-white rounded-xl overflow-hidden border border-slate-800')
                        
                        with ui.row().classes('w-full justify-between mt-6'):
                            ui.button('Vider l\'historique', on_click=lambda: state.compare_table_ui.rows.clear()).classes('bg-red-600/80 rounded-xl')
                            ui.button('Exporter Synthèse (.csv)', on_click=export_comparison_matrix).classes('bg-emerald-600 rounded-xl font-bold')

                # ------------------------------------------
                # ETAPE 4 : PREDICT (INFERENCE)
                # ------------------------------------------
                with ui.tab_panel('t_predict'):
                    with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                        ui.label('Inférence & Prédiction sur Fichiers Vierges').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-4')
                        
                        with ui.row().classes('w-full gap-6'):
                            with ui.card().classes('w-[48%] bg-slate-950/40 p-4 rounded-xl border border-slate-800'):
                                ui.label('1. Fichier d\'Inférence').classes('text-sm font-bold text-slate-300 mb-2')
                                
                                ui.upload(label='Déposez le fichier de test', on_upload=lambda e: import_predict_dataset_from_bytes(e.content.read()), auto_upload=True).classes('w-full rounded-xl')
                                state.predict_file_lbl = ui.label('Aucun fichier d\'inférence chargé').classes('text-slate-400 font-mono text-xs mt-2')
                            
                            with ui.card().classes('w-[48%] bg-slate-950/40 p-4 rounded-xl border border-slate-800'):
                                ui.label('2. Sélection du Cerveau (Modèle)').classes('text-sm font-bold text-slate-300 mb-2')
                                state.predict_run_select = ui.select([], label='Choisir un Run Validé').classes('w-full rounded-xl')
                                ui.button('Synchroniser les modèles', on_click=sync_predict_runs).classes('w-full bg-slate-800 rounded-xl text-xs mt-2')

                        with ui.row().classes('w-full justify-between items-center mt-6 border-t border-slate-800 pt-4'):
                            with ui.row().classes('gap-3'):
                                ui.button('Calculer les prédictions', on_click=execute_inference_process).classes('bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl font-bold')
                                ui.button('Visualiser les estimations', on_click=view_predict_data).classes('bg-blue-600 rounded-xl')
                            ui.button('Exporter le fichier enrichi (.csv)', on_click=export_predicted_csv).classes('bg-indigo-600 rounded-xl')

        # LOGS & DOCS
        with ui.tab_panel('logs'):
            with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                ui.label('Journal d\'Activité Serveur').classes('text-lg font-bold text-emerald-400 mb-4')
                state.logs_area = ui.html().classes('font-mono text-xs bg-black p-4 rounded-xl w-full h-96 overflow-y-scroll text-green-400 border border-slate-800')
                state.logs_area.content = "<br>".join(state.logs)

        with ui.tab_panel('tutorial'):
            with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl text-white'):
                ui.label('Documentation Applicative').classes('text-lg font-bold text-emerald-400 mb-4')
                ui.markdown("L'environnement asynchrone est maintenant 100% blindé face aux téléversements.")

    refresh_algo_param_view()

# ==========================================
# 4. GESTION DU PIPELINE DE LECTURE BINAIRE
# ==========================================

def import_main_dataset_from_bytes(raw_bytes):
    try:
        from io import BytesIO, StringIO
        
        # 1. Détection robuste de l'encodage (UTF-8 ou Latin-1 de secours)
        try:
            text_data = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text_data = raw_bytes.decode('latin-1')
            
        # 2. Détection automatique du séparateur (virgule, point-virgule ou tabulation)
        first_line = text_data.split('\n')[0] if '\n' in text_data else text_data
        sep = ','
        if ';' in first_line and first_line.count(';') > first_line.count(','):
            sep = ';'
        elif '\t' in first_line:
            sep = '\t'
            
        # 3. Chargement dans Pandas via un flux de texte clean
        state.df = pd.read_csv(StringIO(text_data), sep=sep)
        
        # SÉCURITÉ : Nettoyer les noms de colonnes (enlever les espaces ou caractères invisibles)
        state.df.columns = [str(c).strip() for c in state.df.columns]
        
        state.save_state("Import dataset")
        state.log(f"Base chargée avec succès ({len(state.df)} lignes, {len(state.df.columns)} variables). Séparateur détecté : '{sep}'")
        
        # Appeler la synchronisation de l'interface
        sync_all_comboboxes()
    except Exception as ex:
        state.log(f"Échec de chargement : {str(ex)}")
        print(traceback.format_exc()) # Écrit l'erreur complète dans les logs Render pour diagnostic

def import_predict_dataset_from_bytes(raw_bytes):
    try:
        from io import BytesIO, StringIO
        
        try:
            text_data = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text_data = raw_bytes.decode('latin-1')
            
        first_line = text_data.split('\n')[0] if '\n' in text_data else text_data
        sep = ','
        if ';' in first_line and first_line.count(';') > first_line.count(','):
            sep = ';'
        elif '\t' in first_line:
            sep = '\t'
            
        state.df_predict = pd.read_csv(StringIO(text_data), sep=sep)
        state.df_predict.columns = [str(c).strip() for c in state.df_predict.columns]
        
        state.predict_file_lbl.text = f"Fichier inférence validé ({len(state.df_predict)} lignes)"
        state.log("Base d'inférence chargée avec succès.")
    except Exception as ex:
        ui.notify(f"Erreur d'importation : {str(ex)}")
        print(traceback.format_exc())

# ==========================================
# 5. LOGIQUE SECONDAIRE DE NETTOYAGE
# ==========================================

def toggle_split_view(strategy):
    state.split_strategy = strategy
    if strategy == 'Train/Test Split':
        state.split_container_ui.remove_class('hidden')
        state.kfold_container_ui.add_class('hidden')
    else:
        state.split_container_ui.add_class('hidden')
        state.kfold_container_ui.remove_class('hidden')

def sync_all_comboboxes():
    if state.df is None: return
    cols = [str(c) for c in state.df.columns]
    num_cols = [str(c) for c in state.df.select_dtypes(include=[np.number]).columns]
    cat_cols = [str(c) for c in state.df.select_dtypes(include=['object', 'category']).columns]
    
    # 1. Mise à jour des options des menus déroulants
    state.outlier_select.options = num_cols
    if num_cols: state.outlier_select.value = num_cols[0]
    state.outlier_select.update() # <-- CRUCIAL POUR NICEGUI
    
    state.cat_select.options = cat_cols
    if cat_cols: state.cat_select.value = cat_cols[0]
    state.cat_select.update() # <-- CRUCIAL POUR NICEGUI
    
    state.main_target_select.options = cols
    if cols: state.main_target_select.value = cols[0]
    state.main_target_select.update() # <-- CRUCIAL POUR NICEGUI
    
    # 2. Reconstruction et rafraîchissement des cases à cocher (Features)
    state.features_checkbox_container.clear()
    with state.features_checkbox_container:
        for c in cols:
            ui.checkbox(text=c, value=True).classes('text-slate-200 mx-2 font-mono text-sm')
    state.features_checkbox_container.update() # <-- CRUCIAL POUR NICEGUI

def update_cat_reference(e):
    if state.df is None or not e.value: return
    instances = [str(x) for x in state.df[e.value].dropna().unique()]
    state.cat_ref_select.options = instances
    if instances: state.cat_ref_select.value = instances[0]

def process_outliers():
    if state.df is None or not state.outlier_select.value: return
    col = state.outlier_select.value
    mn, mx = state.outlier_min.value, state.outlier_max.value
    if mn is None or mx is None: return ui.notify("Spécifiez les bornes !")
    
    state.save_state(f"Outliers on {col}")
    if 'Clip' in state.outlier_action.value:
        state.df[col] = state.df[col].clip(lower=mn, upper=mx)
    else:
        state.df = state.df[(state.df[col] >= mn) & (state.df[col] <= mx)]
    state.log(f"Filtre Outlier validé sur '{col}'.")
    sync_all_comboboxes()

def run_cat_transformation(action):
    if state.df is None or not state.cat_select.value: return
    col = state.cat_select.value
    state.save_state(f"Cat transform {action} sur {col}")
    
    if action == 'drop':
        state.df.drop(columns=[col], inplace=True)
    else:
        ref = state.cat_ref_select.value
        dummies = pd.get_dummies(state.df[col], prefix=col).astype(int)
        if f"{col}_{ref}" in dummies.columns:
            dummies.drop(columns=[f"{col}_{ref}"], inplace=True)
        state.df = pd.concat([state.df.drop(columns=[col]), dummies], axis=1)
    state.log(f"Transformation de la variable quali '{col}' opérée.")
    sync_all_comboboxes()

def run_scaling_process():
    if state.df is None: return
    scope = state.scale_scope.value
    method = state.scale_method.value
    target = state.main_target_select.value
    
    state.save_state(f"Scale {method}")
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
        state.log(f"Mise à l'échelle accomplie ({method}).")
    except Exception as ex:
        state.log(f"Erreur Scaling : {str(ex)}")

def view_main_data():
    if state.df is None: return ui.notify("Aucune table chargée")
    with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-5xl h-5/6 bg-slate-900 rounded-2xl text-white'):
        ui.label('Aperçu du Dataset principal').classes('text-md font-bold text-emerald-400')
        ui.table(
            columns=[{'name': c, 'label': c, 'field': c} for c in state.df.columns],
            rows=state.df.head(50).to_dict('records')
        ).classes('w-full bg-slate-950 rounded-xl overflow-hidden')
        ui.button('Fermer', on_click=dialog.close).classes('bg-slate-800 rounded-xl self-end')
    dialog.open()

def apply_undo():
    if state.history:
        act, prev = state.history.pop()
        state.future.append((act, state.df.copy()))
        state.df = prev
        state.log(f"[UNDO] Annulation de : {act}")
        sync_all_comboboxes()

def apply_redo():
    if state.future:
        act, nxt = state.future.pop()
        state.history.append((act, state.df.copy()))
        state.df = nxt
        state.log(f"[REDO] Rétablissement de : {act}")
        sync_all_comboboxes()

def refresh_algo_param_view():
    algo = state.algo_choice.value
    state.param_options_frame.clear()
    with state.param_options_frame:
        if algo == 'EcoRETINA':
            state.eco_loss = ui.select(['mse', 'mae', 'MAPE', 'AIC', 'BIC'], value='mse', label='Loss').classes('w-32 rounded-xl')
            state.eco_reg_type = ui.select(['linear', 'logit', 'probit'], value='linear', label='Reg Type').classes('w-32 rounded-xl')
            state.eco_cross_dummy = ui.select(['False', 'True'], value='False', label='Cross Dummy').classes('w-32 rounded-xl')
            state.eco_cov_type = ui.select(['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3'], value='nonrobust', label='Cov Type').classes('w-32 rounded-xl')
            state.eco_grid = ui.number(label='Grid Step', value=0.005, format='%.4f').classes('w-28 rounded-xl')
            state.eco_max_reg = ui.number(label='Max Reg', value=100).classes('w-24 rounded-xl')
            state.eco_chunk_size = ui.number(label='Chunk Size', value=500).classes('w-24 rounded-xl')
            state.eco_seed = ui.number(label='Seed', value=8).classes('w-20 rounded-xl')
            
        elif algo == 'OLS':
            state.ols_fit_intercept = ui.select(['True', 'False'], value='True', label='Fit Intercept').classes('w-40 rounded-xl')
            
        elif algo in ['Lasso', 'Ridge', 'ElasticNet']:
            state.alpha_input = ui.number(label='Alpha (Penalty)', value=0.01, format='%.4f').classes('w-40 rounded-xl')
            state.fit_intercept_input = ui.select(['True', 'False'], value='True', label='Fit Intercept').classes('w-40 rounded-xl')
            state.max_iter_input = ui.number(label='Max Iterations', value=1000).classes('w-40 rounded-xl')
            state.tol_input = ui.number(label='Tolerance', value=0.0001, format='%.4f').classes('w-40 rounded-xl')
            if algo == 'Ridge':
                state.ridge_solver = ui.select(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], value='auto', label='Solver').classes('w-40 rounded-xl')
            if algo == 'ElasticNet':
                state.en_l1_ratio = ui.number(label='L1 Ratio', value=0.5, format='%.2f').classes('w-40 rounded-xl')
                
        elif algo == 'XGBoost':
            state.xgb_n = ui.number(label='N Estimators', value=100).classes('w-40 rounded-xl')
            state.xgb_depth = ui.number(label='Max Depth', value=6).classes('w-40 rounded-xl')
            state.xgb_lr = ui.number(label='Learning Rate', value=0.1, format='%.3f').classes('w-40 rounded-xl')
            state.xgb_subsample = ui.number(label='Subsample', value=1.0, format='%.2f').classes('w-40 rounded-xl')
            state.xgb_colsample = ui.number(label='Colsample By Tree', value=1.0, format='%.2f').classes('w-40 rounded-xl')
            state.xgb_gamma = ui.number(label='Gamma', value=0.0).classes('w-40 rounded-xl')
            state.xgb_alpha = ui.number(label='Reg Alpha (L1)', value=0.0).classes('w-40 rounded-xl')
            state.xgb_lambda = ui.number(label='Reg Lambda (L2)', value=1.0).classes('w-40 rounded-xl')
            
        elif algo == 'Random Forest':
            state.rf_n_estimators = ui.number(label='N Estimators', value=100).classes('w-40 rounded-xl')
            state.rf_max_depth = ui.number(label='Max Depth (0=Unl.)', value=0).classes('w-40 rounded-xl')
            state.rf_min_split = ui.number(label='Min Samples Split', value=2).classes('w-40 rounded-xl')
            state.rf_min_leaf = ui.number(label='Min Samples Leaf', value=1).classes('w-40 rounded-xl')
            state.rf_max_features = ui.select(['1.0', 'sqrt', 'log2'], value='1.0', label='Max Features').classes('w-40 rounded-xl')
            
        elif algo == 'Neural Network':
            state.nn_layers = ui.input(label='Hidden Layers', value="100,50").classes('w-40 rounded-xl')
            state.nn_activation = ui.select(['relu', 'tanh', 'logistic', 'identity'], value='relu', label='Activation').classes('w-40 rounded-xl')
            state.nn_solver = ui.select(['adam', 'sgd', 'lbfgs'], value='adam', label='Solver').classes('w-40 rounded-xl')
            state.nn_alpha = ui.number(label='Alpha (L2 Penalty)', value=0.0001, format='%.4f').classes('w-40 rounded-xl')
            state.nn_lr_init = ui.number(label='Learning Rate Init', value=0.001, format='%.3f').classes('w-40 rounded-xl')
            state.nn_max_iter = ui.number(label='Max Iterations', value=200).classes('w-40 rounded-xl')
            
    state.param_options_frame.update() # <--- Force NiceGUI à redessiner l'interface web instantanément
async def trigger_pipeline_execution():
    if state.df is None: return ui.notify("Aucune base exploitable !")
    target = state.main_target_select.value
    algo = state.algo_choice.value
    
    selected_features = []
    for child in state.features_checkbox_container:
        if isinstance(child, ui.checkbox) and child.value and child.text != target:
            selected_features.append(child.text)
            
    if not selected_features: return ui.notify("Cochez au moins une variable explicative !")

    state.algo_status_lbl.text = "Exécution mathématique de la régression en tâche de fond..."
    state.algo_status_lbl.update()
    
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

    res = await run.cpu_bound(execute_ml_math_core, algo, target, selected_features, config_args)
    
    run_id = f"Run_{time.strftime('%H%M%S')}"
    state.run_history[run_id] = res
    state.latest_run_by_algo[algo] = run_id
    
    state.compare_table_ui.add_rows([{
        'run': run_id, 'algo': algo,
        'r2_tr': f"{res['metrics']['r2_tr']:.4f}", 'mape_tr': f"{res['metrics']['mape_tr']:.2f}%",
        'r2_te': f"{res['metrics']['r2_te']:.4f}", 'rmse_te': f"{res['metrics']['rmse_te']:.4f}",
        'mape_te': f"{res['metrics']['mape_te']:.2f}%", 'co2': f"{res['metrics']['emissions']:.5f}"
    }])
    
    state.algo_status_lbl.text = f"Modèle sauvegardé : {run_id}"
    state.log(f"Pipeline complété avec succès pour {algo}.")

def execute_ml_math_core(algo, target, features, args):
    df_clean = state.df.dropna(subset=[target] + features).fillna(0)
    X = df_clean[features].values
    y = df_clean[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tracker = EmissionsTracker(tracking_mode='process', log_level='error')
    tracker.start()
    
    model = None
    feature_names_final = features.copy()
    
    # Extraction des booléens textuels
    fit_intercept_bool = True if str(args.get('fit_intercept', 'True')) == 'True' else False
    cross_dummy_bool = True if str(args.get('eco_cross_dummy', 'False')) == 'True' else False

    if algo == 'OLS':
        if fit_intercept_bool:
            X_train_fit = sm.add_constant(X_train, has_constant='add')
            X_test_fit = sm.add_constant(X_test, has_constant='add')
            feature_names_final = ['const'] + features
        else:
            X_train_fit, X_test_fit = X_train, X_test
        sm_res = sm.OLS(y_train, pd.DataFrame(X_train_fit, columns=feature_names_final)).fit(cov_type=args.get('eco_cov_type', 'nonrobust'))
        model = OLSWrapper(sm_res)
        y_train_pred = sm_res.predict(pd.DataFrame(X_train_fit, columns=feature_names_final))
        y_test_pred = sm_res.predict(pd.DataFrame(X_test_fit, columns=feature_names_final))
        
    elif algo in ['Lasso', 'Ridge', 'ElasticNet']:
        if algo == 'Lasso':
            model_pen = Lasso(alpha=args['alpha'], fit_intercept=fit_intercept_bool, max_iter=args['max_iter'], tol=args['tol'])
        elif algo == 'Ridge':
            model_pen = Ridge(alpha=args['alpha'], fit_intercept=fit_intercept_bool, max_iter=args['max_iter'], tol=args['tol'], solver=args['ridge_solver'])
        else:
            model_pen = ElasticNet(alpha=args['alpha'], l1_ratio=args['en_l1_ratio'], fit_intercept=fit_intercept_bool, max_iter=args['max_iter'], tol=args['tol'])
        
        model_pen.fit(X_train, y_train)
        sel_idx = np.where(np.abs(model_pen.coef_) > 1e-5)[0]
        if len(sel_idx) == 0: sel_idx = np.arange(X_train.shape[1])
        
        X_tr_sel, X_te_sel = X_train[:, sel_idx], X_test[:, sel_idx]
        feature_names_final = [features[i] for i in sel_idx]
        
        if fit_intercept_bool:
            X_tr_fit = sm.add_constant(X_tr_sel, has_constant='add')
            X_te_fit = sm.add_constant(X_te_sel, has_constant='add')
            feature_names_final = ['const'] + feature_names_final
        else:
            X_tr_fit, X_te_fit = X_tr_sel, X_te_sel
            
        sm_res = sm.OLS(y_train, pd.DataFrame(X_tr_fit, columns=feature_names_final)).fit(cov_type=args.get('eco_cov_type', 'nonrobust'))
        model = OLSWrapper(sm_res)
        y_train_pred = sm_res.predict(pd.DataFrame(X_tr_fit, columns=feature_names_final))
        y_test_pred = sm_res.predict(pd.DataFrame(X_te_fit, columns=feature_names_final))
        
    elif algo == 'XGBoost':
        model = xgb.XGBRegressor(
            n_estimators=args['xgb_n'], max_depth=args['xgb_depth'], learning_rate=args['xgb_lr'],
            subsample=args['xgb_subsample'], colsample_bytree=args['xgb_colsample'], gamma=args['xgb_gamma'],
            reg_alpha=args['xgb_alpha'], reg_lambda=args['xgb_lambda'], random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        
    elif algo == 'Random Forest':
        depth = None if args['rf_max_depth'] == 0 else args['rf_max_depth']
        max_f = None if args['rf_max_features'] == '1.0' else args['rf_max_features']
        model = RandomForestRegressor(
            n_estimators=args['rf_n_estimators'], max_depth=depth, 
            min_samples_split=args['rf_split'], min_samples_leaf=args['rf_leaf'], max_features=max_f, random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        
    elif algo == 'Neural Network':
        layers = tuple(int(x.strip()) for x in args['nn_layers'].split(','))
        model = MLPRegressor(
            hidden_layer_sizes=layers, activation=args['nn_activation'], solver=args['nn_solver'],
            alpha=args['nn_alpha'], learning_rate_init=args['nn_lr_init'], max_iter=args['nn_max_iter'], random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        
    elif algo == 'EcoRETINA':
        # Intégration de la structure eco_retina si disponible
        if ECO_RETINA_AVAILABLE:
            model = EcoRETINA()
            # Simulation/Passage identique à ta configuration originale de eco_retina.py
            model.fit(y=y_train, X=X_train, col_names=features, loss=args['eco_loss'], grid=args['eco_grid'], reg_type=args['eco_reg_type'], cross_dummy=cross_dummy_bool, max_reg=args['eco_max_reg'], chunk_size=args['eco_chunk_size'], seed=args['eco_seed'], cov_type=args['eco_cov_type'])
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            # Fallback de secours si eco_retina.py est absent du dépôt
            model = Ridge(alpha=0.1)
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
def open_detailed_report():
    algo = state.algo_choice.value
    run_id = state.latest_run_by_algo.get(algo)
    if not run_id: return ui.notify("Aucune analyse disponible pour cet algorithme !")
    
    run_data = state.run_history[run_id]
    y_test = run_data['y_test']
    y_test_pred = run_data['y_test_pred']
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#0f172a')
    
    axes[0].scatter(y_test, y_test_pred, alpha=0.6, color='#38bdf8')
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='#34d399', linestyle='--')
    axes[0].set_title('Reals vs Predictions', color='white', fontsize=10)
    axes[0].set_facecolor('#1e293b')
    axes[0].tick_params(colors='white')
    
    residuals = y_test - y_test_pred
    axes[1].hist(residuals, bins=15, color='#f87171', alpha=0.8)
    axes[1].set_title('Residuals Split', color='white', fontsize=10)
    axes[1].set_facecolor('#1e293b')
    axes[1].tick_params(colors='white')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-4xl bg-slate-900 rounded-2xl text-white p-6 border border-slate-800'):
        ui.label(f"Indicateurs Analytiques - {run_id}").classes('text-lg font-black text-emerald-400')
        ui.html(f'<div class="flex justify-center mt-2 rounded-xl overflow-hidden border border-slate-800"><img src="data:image/png;base64,{img_b64}"/></div>')
        ui.button('Fermer le rapport', on_click=dialog.close).classes('bg-slate-800 rounded-xl mt-4 self-end')
    dialog.open()

def export_comparison_matrix():
    if not state.compare_table_ui.rows: return ui.notify("Aucune donnée disponible")
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Run_ID', 'Algorithm', 'R2_Train', 'R2_Test', 'MAPE_Test', 'CO2_kg'])
    for r in state.compare_table_ui.rows:
        writer.writerow([r['run'], r['algo'], r['r2_tr'], r['r2_te'], r['mape_te'], r['co2']])
    ui.download(output.getvalue().encode('utf-8'), 'ML_Workbench_Benchmark.csv')

def sync_predict_runs():
    runs = list(state.run_history.keys())
    state.predict_run_select.options = runs
    if runs: state.predict_run_select.value = runs[-1]

def execute_inference_process():
    if state.df_predict is None: return ui.notify("Importez d'abord une base !")
    run_id = state.predict_run_select.value
    if not run_id: return ui.notify("Sélectionnez un modèle ajusté !")
    
    run_data = state.run_history[run_id]
    raw_feats = run_data['raw_features']
    model = run_data['model']
    feature_names = run_data['feature_names']
    target_col = run_data['target_col']
    
    try:
        X_new = state.df_predict[raw_feats].fillna(0).values
        if 'const' in feature_names: X_new = sm.add_constant(X_new, has_constant='add')
        preds = model.predict(pd.DataFrame(X_new, columns=feature_names))
        pred_col = f"Predicted_{target_col}_{run_id}"
        state.df_predict[pred_col] = preds
        ui.notify(f"Prédictions générées dans '{pred_col}' !")
    except Exception as ex:
        ui.notify(f"Erreur mathématique lors du calcul : {str(ex)}")

def view_predict_data():
    if state.df_predict is None: return ui.notify("Aucune base d'inférence")
    with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-5xl h-5/6 bg-slate-900 rounded-2xl text-white'):
        ui.label('Vérification des Prédictions Appliquées').classes('text-md font-bold text-emerald-400')
        ui.table(
            columns=[{'name': c, 'label': c, 'field': c} for c in state.df_predict.columns],
            rows=state.df_predict.head(50).to_dict('records')
        ).classes('w-full bg-slate-950 rounded-xl overflow-hidden')
        ui.button('Fermer', on_click=dialog.close).classes('bg-slate-800 rounded-xl self-end')
    dialog.open()

def export_predicted_csv():
    if state.df_predict is None: return
    csv_buf = StringIO()
    state.df_predict.to_csv(csv_buf, index=False)
    ui.download(csv_buf.getvalue().encode('utf-8'), 'EcoRETINA_Predictions_Output.csv')

# Liaison réseau définitive sur le port public attribué par Render
ui.run(port=int(os.environ.get('PORT', 8080)), title="EcoRETINA Intelligence", reload=False)
