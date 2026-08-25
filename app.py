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
import matplotlib
matplotlib.use('Agg')  # Anti-crash thread-safe backend
import matplotlib.pyplot as plt
import io
import base64

from nicegui import ui, run, app
from codecarbon import EmissionsTracker

# --- ALGORITHMES ET IMPORTS MACHINE LEARNING ---
import xgboost as xgb
import requests

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ==========================
# Summary rapport generation and visualization
# =========================

def generate_summary_plots_base64(run_data):
    # Conversion forcée en numpy array pour permettre les opérations mathématiques vectorielles
    y_test = np.asarray(run_data['y_test'], dtype=np.float64)
    y_test_pred = np.asarray(run_data['y_test_pred'], dtype=np.float64)
    
    metrics = run_data['metrics']
    model_name = run_data['model_name']
    data_rows = run_data.get('data_rows', [])
    elasticity_data = run_data.get('elasticity_data', [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0f172a')
    fig.tight_layout(pad=4.0)
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_test_pred, alpha=0.6, color='#38bdf8', label='Points')
    min_v, max_v = float(np.min(y_test)), float(np.max(y_test))
    axes[0].plot([min_v, max_v], [min_v, max_v], color='#22c55e', linestyle='--', label='Fit idéal')
    std_resid = np.std(y_test - y_test_pred)
    x_line = np.linspace(min_v, max_v, 100)
    axes[0].fill_between(x_line, x_line - 1.96 * std_resid, x_line + 1.96 * std_resid, color='#22c55e', alpha=0.15, label='95% CI')
    axes[0].set_title('Actual vs Predicted (Test Set)', color='white')
    axes[0].set_xlabel('Actual', color='white')
    axes[0].set_ylabel('Predicted', color='white')
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor('#1e293b')
    axes[0].legend(facecolor='#0f172a', edgecolor='none', labelcolor='white', loc='upper left', fontsize=8)

    # 2. Residuals Distribution
    residuals = y_test - y_test_pred
    axes[1].hist(residuals, bins=20, color='#ef4444', alpha=0.7, edgecolor='white')
    axes[1].set_title('Residuals Distribution', color='white')
    axes[1].set_xlabel('Residual', color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor('#1e293b')
    p_val = metrics.get('norm_p_val', float('nan'))
    axes[1].text(0.95, 0.95, f"P-Value: {p_val:.4f}" if not np.isnan(p_val) else "P-Value: N/A", 
                 transform=axes[1].transAxes, color='white', fontsize=10, ha='right', va='top', 
                 bbox=dict(facecolor='#0f172a', alpha=0.8, edgecolor='none'))

    # 3. Top Features / Elasticities
    plot_data = [d for d in data_rows if d["Variable"] != "const"]
    val_key = "Importance" if len(plot_data) > 0 and "Importance" in plot_data[0] else "Coefficient" if len(plot_data) > 0 else None
        
    if val_key:
        top_feats = sorted(plot_data, key=lambda x: abs(x[val_key]), reverse=True)[:10]
        y_labels = [(d["Variable"][:15] + "..") if len(d["Variable"]) > 15 else d["Variable"] for d in top_feats][::-1]
        x_vals = [d[val_key] for d in top_feats][::-1]
        colors = ['#22c55e' if v >= 0 else '#ef4444' for v in x_vals]
        axes[2].barh(y_labels, x_vals, color=colors)
        axes[2].set_title(f"Top 10 Variables ({model_name})", color='white')
        axes[2].tick_params(colors='white')
        axes[2].set_facecolor('#1e293b')
    elif model_name == 'EcoRETINA' and len(elasticity_data) > 0:
        top_elast = sorted(elasticity_data, key=lambda x: abs(x["Elasticity"]), reverse=True)[:10]
        y_labels = [(d["Variable"][:15] + "..") if len(d["Variable"]) > 15 else d["Variable"] for d in top_elast][::-1]
        e_vals = [d["Elasticity"] for d in top_elast][::-1]
        colors = ['#22c55e' if v >= 0 else '#ef4444' for v in e_vals]
        axes[2].barh(y_labels, e_vals, color=colors)
        axes[2].axvline(x=0, color='white', linestyle=':', alpha=0.6)
        axes[2].set_title('Top 10 Elasticities (%)', color='white')
        axes[2].set_xlabel('% Change in Y for +1% in X', color='white')
        axes[2].tick_params(colors='white')
        axes[2].set_facecolor('#1e293b')
    else:
        axes[2].text(0.5, 0.5, "No Feature Data Available", color='white', ha='center', va='center')
        axes[2].set_facecolor('#1e293b')
        axes[2].axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)
    del fig, axes
    import gc
    gc.collect()
    
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def handle_table_row_click(event, state):
    args = getattr(event, 'args', None)
    run_id = None

    # Quasar transmet généralement [event_js, row_data, index]
    if isinstance(args, list) and len(args) >= 2 and isinstance(args[1], dict):
        run_id = args[1].get('run')
    elif isinstance(args, dict):
        run_id = args.get('run')
    elif hasattr(event, 'row') and isinstance(event.row, dict):
        run_id = event.row.get('run')

    if run_id:
        clean_id = str(run_id).replace('⭐', '').replace('★', '').strip()
        if clean_id in state.run_history:
            show_summary_dialog(clean_id, state)
        else:
            ui.notify(f"Run {clean_id} not found in history.", type='warning')

def show_summary_dialog(run_id, state):
    run_data = state.run_history.get(run_id)
    if not run_data:
        return ui.notify(f"Run {run_id} not found.", type='warning')

    model = run_data.get("model")
    model_name = run_data.get("model_name", "Unknown")
    feature_names = run_data.get("feature_names", [])
    raw_features = run_data.get("raw_features", [])
    metrics = run_data.get("metrics", {})
    y_test = run_data.get("y_test", [])
    y_test_pred = run_data.get("y_test_pred", [])

    summary_lines = []
    data_rows = []
    elasticity_data = []

    # 1. Performance Metrics
    summary_lines.append("=" * 110)
    summary_lines.append(f"{'PIPELINE PERFORMANCE METRICS':^110}")
    summary_lines.append("=" * 110)
    summary_lines.append(f" TRAIN -> R-squared: {metrics.get('r2_tr', 0):>8.4f} | Adj R2: {metrics.get('adj_r2_tr', 0):>8.4f} | RMSE: {metrics.get('rmse_tr', 0):>8.4f} | MAPE: {metrics.get('mape_tr', 0):>6.2f}%")
    summary_lines.append(f" TEST  -> R-squared: {metrics.get('r2_te', 0):>8.4f} | Adj R2: {metrics.get('adj_r2_te', 0):>8.4f} | RMSE: {metrics.get('rmse_te', 0):>8.4f} | MAPE: {metrics.get('mape_te', 0):>6.2f}%")
    summary_lines.append(f" Residuals Normality (Shapiro-Wilk) : P-Value = {metrics.get('norm_p_val', 0):.6f}")
    summary_lines.append(f" Emissions (kgCO2eq)                 : {metrics.get('emissions', 0):.8f}\n")

    # 2. Elasticities (EcoRETINA)
    if model_name == 'EcoRETINA' and hasattr(model, 'predict'):
        summary_lines.append("=" * 110)
        summary_lines.append(f"{'ECONOMETRIC ELASTICITIES & MARGINAL EFFECTS AT MEAN POINT':^110}")
        summary_lines.append("=" * 110)
        summary_lines.append(f" {'Original Variable':<30} | {'Mean Value (X)':>18} | {'Marginal Effect (dY/dX)':>24} | {'Elasticity (%)':>18}")
        summary_lines.append("-" * 110)

        con_indices = getattr(model, 'con_cols_indices', [])
        X_raw_test = run_data.get("X_test", np.zeros((1, len(raw_features))))
        
        if con_indices and len(X_raw_test) > 0:
            X_mean_raw = np.mean(X_raw_test, axis=0, keepdims=True)
            y_mean = np.mean(y_test) if np.mean(y_test) != 0 else 1e-9
            y_base = model.predict(X_mean_raw)[0]
            delta = 1e-4

            for idx in con_indices:
                feat_label = raw_features[idx] if idx < len(raw_features) else f"Var_{idx}"
                x_val = X_mean_raw[0, idx]
                h = x_val * delta if x_val != 0 else delta
                X_pert = X_mean_raw.copy()
                X_pert[0, idx] += h
                try:
                    y_pert = model.predict(X_pert)[0]
                    marginal_effect = (y_pert - y_base) / h
                    elasticity = marginal_effect * (x_val / y_mean)
                except Exception:
                    marginal_effect, elasticity = 0.0, 0.0

                summary_lines.append(f" {feat_label[:30]:<30} | {x_val:>18.4f} | {marginal_effect:>24.6f} | {elasticity:>17.4f}%")
                elasticity_data.append({"Variable": feat_label, "Elasticity": elasticity, "MarginalEffect": marginal_effect})
        summary_lines.append("\n")

    # 3. Features & Coefficients
    if model_name in ['EcoRETINA', 'OLS', 'Lasso', 'Ridge', 'ElasticNet'] and (hasattr(model, 'sm_model') or hasattr(model, 'coef_')):
        summary_lines.append("=" * 110)
        summary_lines.append(f"{'ENGINEERED FEATURES & STATISTICAL SIGNIFICANCE':^110}")
        summary_lines.append("=" * 110)
        summary_lines.append(f" {'Variable':<30} | {'Coefficient':>22} | {'T-Stat':>15} | {'P-Value':>15}")
        summary_lines.append("-" * 110)

        is_sm = hasattr(model, 'sm_model')
        params_dict = model.sm_model.params if is_sm else dict(zip(feature_names, model.coef_))
        sorted_feats = sorted(feature_names, key=lambda f: abs(params_dict.get(f, 0)), reverse=True)

        for feat in sorted_feats:
            coef = params_dict.get(feat, 0.0)
            tstat = model.sm_model.tvalues.get(feat, float('nan')) if is_sm else float('nan')
            pval = model.sm_model.pvalues.get(feat, float('nan')) if is_sm else float('nan')
            summary_lines.append(f" {feat[:30]:<30} | {coef:>22.6f} | {tstat:>15.3f} | {pval:>15.4f}")
            data_rows.append({"Variable": feat, "Coefficient": coef, "T-Stat": tstat, "P-Value": pval})
    else:
        summary_lines.append("=" * 110)
        summary_lines.append(f"{'FEATURE IMPORTANCES (TREE-BASED / NEURAL NETWORKS)':^110}")
        summary_lines.append("=" * 110)
        importances = []
        if model_name == 'Neural Network' and hasattr(model, 'coefs_'):
            importances = np.mean(np.abs(model.coefs_[0]), axis=1)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        importances = np.array(importances).flatten()
        for feat, val in sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True):
            summary_lines.append(f" {feat[:35]:<35} | {float(val):>25.6f}")
            data_rows.append({"Variable": feat, "Importance": val})

    run_data['data_rows'] = data_rows
    run_data['elasticity_data'] = elasticity_data

    # Formule mathématique
    eq = "Y = \n"
    if hasattr(model, 'sm_model') or hasattr(model, 'coef_'):
        params = model.sm_model.params if hasattr(model, 'sm_model') else dict(zip(feature_names, model.coef_))
        intercept = getattr(model, 'intercept_', 0.0)
        if intercept != 0.0:
            eq += f"   {intercept:.4f}\n"
        for f in feature_names:
            c = params.get(f, 0)
            if c != 0:
                eq += f"   {'+' if c > 0 else '-'} {abs(c):.4f} * ({f})\n"
    else:
        eq += "[Mathematical equation not applicable for Tree-based or Neural Network algorithms.]"

    # Construction de la modale d'affichage NiceGUI
    with ui.dialog() as dialog, ui.card().classes('w-[1100px] max-w-7xl max-h-[90vh] bg-slate-900 border border-slate-700 text-slate-100 flex flex-col'):
        with ui.row().classes('w-full justify-between items-center pb-2 border-b border-slate-800'):
            ui.label(f"📊 Detailed Analysis - {run_id} ({model_name})").classes('text-lg font-bold text-emerald-400')
            ui.button(icon='close', on_click=dialog.close).props('flat round dense').classes('text-slate-400 hover:text-white')

        with ui.tabs().classes('w-full text-slate-300') as tabs:
            t1 = ui.tab('Statistical Report')
            t2 = ui.tab('Visual Formula')
            t3 = ui.tab('Graphical Analysis')

        with ui.tab_panels(tabs, value=t1).classes('w-full bg-slate-950/60 p-4 rounded-xl flex-grow overflow-auto'):
            with ui.tab_panel(t1):
                ui.code("\n".join(summary_lines)).classes('w-full text-xs font-mono bg-transparent text-emerald-300')
            
            with ui.tab_panel(t2):
                ui.code(eq).classes('w-full text-sm font-mono bg-transparent text-cyan-300')

            with ui.tab_panel(t3):
                b64_img = generate_summary_plots_base64(run_data)
                ui.image(f'data:image/png;base64,{b64_img}').classes('w-full rounded-lg shadow-xl')

    dialog.open()

@app.head('/')
def read_head():
    return {"status": "ok"}

import traceback

try:
    from eco_retina_V3 import EcoRETINA
    ECO_RETINA_AVAILABLE = True
    print("[INFO] EcoRETINA V3 importé avec succès !")
except Exception as e:
    ECO_RETINA_AVAILABLE = False
    print(f"[ERREUR CRITIQUE] Échec import EcoRETINA V3 : {e}")
    traceback.print_exc()

warnings.filterwarnings("ignore")

# ==========================================
# 1. ARCHITECTURE DES DONNÉES ET ÉTAT LOCAL
# ==========================================
import gc

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
        self.star_phase = 0

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        self.logs.append(log_line)
        ui.notify(message)
        if hasattr(self, 'logs_area') and self.logs_area:
            self.logs_area.content = "<br>".join(self.logs[-50:])  # Garde seulement les 50 derniers logs

    def save_state(self, action_name):
        if self.df is not None:
            # Réduit à 3 étapes de Undo pour économiser la RAM
            self.history.append((action_name, self.df.copy(deep=True)))
            if len(self.history) > 3:
                self.history.pop(0)
            self.future.clear()
            gc.collect()


class OLSWrapper:
    def __init__(self, res): 
        self.sm_model = res
    def predict(self, X): 
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.sm_model.predict(X)

# ==========================================
# 2. IA COPILOT MULTI-PROVIDER
# ==========================================
import json
import urllib.request
import asyncio

class EcoRetinaChatAgent:
    def __init__(self, api_key: str, provider: str):
        self.provider = provider
        self.api_key = api_key.strip()
        self.system_prompt = (
            "You are the Chief Econometrician and AI Support Guide for the EcoRETINA ML Workbench.\n\n"
            "ROLE 1 - WORKBENCH NAVIGATOR: Guide users if they are lost.\n"
            "- Tab 1 (Data): Import, handle outliers, encode dummies, drop columns, scale data.\n"
            "- Tab 2 (Algorithms): Select model, set hyperparameters, run training pipeline.\n"
            "- Tab 3 (Compare): Compare runs, analyze stats, view historical benchmarks.\n"
            "- Tab 4 (Predict): Load new datasets for Inference and export predictions.\n\n"
            "ROLE 2 - SENIOR ECONOMETRICIAN: When analyzing run metrics, provide a rigorous academic interpretation:\n"
            "1. Assess R-squared and Adjusted R-squared to explain the variance captured.\n"
            "2. Identify potential OVERFITTING by strictly comparing Train vs Test performance.\n"
            "3. Evaluate prediction accuracy using RMSE and MAPE.\n"
            "4. Analyze the Shapiro-Wilk p-value (if > 0.05, residuals are normally distributed, validating hypotheses).\n"
            "5. Comment on environmental efficiency based on CodeCarbon emissions (kgCO2eq).\n\n"
            "ROLE 3 - ACTIVE DATA ENGINEER (CRITICAL): If the user explicitly asks you to modify the dataset "
            "(drop columns, drop missing values, or encode variables), you MUST execute it by adding exact command tags at the very end of your response.\n"
            "Available tags:\n"
            "[CMD_DROP_COL:column_name] -> Drops the specified column.\n"
            "[CMD_DROP_NA] -> Drops all rows with missing values.\n"
            "[CMD_ENCODE:column_name] -> One-Hot Encodes the categorical column.\n"
            "Example response: 'I have deleted the ID column as requested. [CMD_DROP_COL:ID]'\n\n"
            "TONE: Professional, pedagogical, highly structured. Answer in English.\n"
            "CRITICAL FORMATTING RULE: DO NOT use any Markdown or LaTeX formatting (no asterisks, no underscores, no bold). "
            "Output strictly plain text. Use ALL CAPS for emphasis and simple hyphens (-) for lists."
        )
        self.history = []

    def _execute_raw_http(self, text: str) -> str:
        headers = {"Content-Type": "application/json"}

        # 1. GOOGLE GEMINI (REST v1beta)
        if self.provider == "Google Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
            
            gemini_contents = []
            for h in self.history:
                role = "user" if h["role"] == "user" else "model"
                gemini_contents.append({"role": role, "parts": [{"text": h["content"]}]})
            gemini_contents.append({"role": "user", "parts": [{"text": text}]})

            payload = {
                "system_instruction": {"parts": [{"text": self.system_prompt}]},
                "contents": gemini_contents
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["candidates"][0]["content"]["parts"][0]["text"]
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
                return reply

        # 2. OPENAI (ChatGPT)
        elif self.provider == "OpenAI (ChatGPT)":
            url = "https://api.openai.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
            messages = [{"role": "system", "content": self.system_prompt}] + self.history + [{"role": "user", "content": text}]
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["choices"][0]["message"]["content"]
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
                return reply

        # 3. GROQ
        elif self.provider == "Groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
            messages = [{"role": "system", "content": self.system_prompt}] + self.history + [{"role": "user", "content": text}]
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["choices"][0]["message"]["content"]
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
                return reply

        # 4. CLAUDE (ANTHROPIC)
        elif self.provider == "Claude (Anthropic)":
            url = "https://api.anthropic.com/v1/messages"
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            messages = self.history + [{"role": "user", "content": text}]
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "system": self.system_prompt,
                "messages": messages
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["content"][0]["text"]
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
                return reply

        return "Unknown AI Provider selected."

    async def ask(self, text: str, bubble_ui):
        max_retries = 3
        loop = asyncio.get_event_loop()
        
        for attempt in range(max_retries):
            try:
                reply = await loop.run_in_executor(None, self._execute_raw_http, text)
                bubble_ui.text = reply
                bubble_ui.update()
                return

            except Exception as e:
                err_msg = str(e)
                if ("503" in err_msg or "unavailable" in err_msg.lower() or "429" in err_msg) and attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    bubble_ui.text = f"⏳ Service busy ({err_msg}). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})"
                    bubble_ui.update()
                    await asyncio.sleep(wait_time)
                    continue

                bubble_ui.text = f"[API Error] : {err_msg}"
                bubble_ui.update()
                return

# ==========================================
# 3. MOTEURS STATELESS (LECTURE & CALCULS ML)
# ==========================================
def _parse_csv_bytes(raw_bytes):
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

    df = pd.read_csv(io.StringIO(text_data), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def execute_ml_math_core(df_input, algo, target, features, args):
    import gc
    tracker = None
    emissions = np.nan
    try:
        tracker = EmissionsTracker(
            tracking_mode='process',
            save_to_file=False,
            log_level='error'
        )
        tracker.start()


    except Exception as e:
        print(f"[CODECARBON WARNING] Tracker start skipped: {e}")
    
    # 1. Sélection numérique stricte en float32
    df_subset = df_input[[target] + list(features)].apply(pd.to_numeric, errors='coerce')
    df_clean = df_subset.dropna(subset=[target]).fillna(0)
    del df_subset

    # Garde-fou Cloud : max 4000 lignes pour EcoRETINA
    if algo == 'EcoRETINA' and len(df_clean) > 4000:
        df_clean = df_clean.sample(n=4000, random_state=42)

    y = np.ascontiguousarray(df_clean[target].values, dtype=np.float32)
    X_encoded = np.ascontiguousarray(df_clean[features].values, dtype=np.float32)
    del df_clean
    gc.collect()

    split_ratio = float(args.get('split_ratio', 0.8))
    if split_ratio >= 1.0:
        X_train, X_test, y_train, y_test = X_encoded, X_encoded, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, train_size=split_ratio, random_state=42
        )
    del X_encoded
    gc.collect()

    model = None
    feature_names_final = list(features)
    fit_intercept_bool = True if str(args.get('fit_intercept', 'True')).lower() in ['true', '1'] else False

    # --- 1. OLS ---
    if algo == 'OLS':
        if fit_intercept_bool:
            X_train_fit = sm.add_constant(X_train, has_constant='add')
            X_test_fit = sm.add_constant(X_test, has_constant='add')
            feature_names_final = ['const'] + list(features)
        else:
            X_train_fit, X_test_fit = X_train, X_test
        
        sm_res = sm.OLS(y_train, X_train_fit).fit(cov_type=args.get('eco_cov_type', 'nonrobust'))
        model = OLSWrapper(sm_res)
        y_train_pred = sm_res.predict(X_train_fit)
        y_test_pred = sm_res.predict(X_test_fit)
        del X_train_fit, X_test_fit

    # --- 2. PENALIZED (Lasso, Ridge, ElasticNet) ---
    elif algo in ['Lasso', 'Ridge', 'ElasticNet']:
        if algo == 'Lasso':
            model = Lasso(alpha=float(args['alpha']), fit_intercept=fit_intercept_bool, max_iter=int(args['max_iter']), tol=float(args['tol']))
        elif algo == 'Ridge':
            model = Ridge(alpha=float(args['alpha']), fit_intercept=fit_intercept_bool, max_iter=int(args['max_iter']), tol=float(args['tol']), solver=args.get('ridge_solver', 'auto'))
        else:
            model = ElasticNet(alpha=float(args['alpha']), l1_ratio=float(args['en_l1_ratio']), fit_intercept=fit_intercept_bool, max_iter=int(args['max_iter']), tol=float(args['tol']))
        
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # --- 3. XGBOOST ---
    elif algo == 'XGBoost':
        model = xgb.XGBRegressor(
            n_estimators=int(args.get('xgb_n', 100)),
            max_depth=int(args.get('xgb_depth', 6)),
            learning_rate=float(args.get('xgb_lr', 0.1)),
            subsample=float(args.get('xgb_subsample', 1.0)),
            colsample_bytree=float(args.get('xgb_colsample', 1.0)),
            gamma=float(args.get('xgb_gamma', 0.0)),
            reg_alpha=float(args.get('xgb_alpha', 0.0)),
            reg_lambda=float(args.get('xgb_lambda', 1.0)),
            n_jobs=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # --- 4. RANDOM FOREST ---
    elif algo == 'Random Forest':
        raw_depth = args.get('rf_max_depth', 12)
        depth = None if (raw_depth is None or int(raw_depth) == 0) else int(raw_depth)
        max_f = None if str(args.get('rf_max_features', '1.0')) == '1.0' else args.get('rf_max_features')
        model = RandomForestRegressor(
            n_estimators=int(args.get('rf_n_estimators', 100)),
            max_depth=depth,
            min_samples_split=int(args.get('rf_split', 2)),
            min_samples_leaf=int(args.get('rf_leaf', 1)),
            max_features=max_f,
            n_jobs=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # --- 5. NEURAL NETWORK ---
    elif algo == 'Neural Network':
        layers = tuple(int(x.strip()) for x in str(args.get("nn_layers", "100,50")).split(','))
        model = MLPRegressor(
            hidden_layer_sizes=layers,
            activation=str(args.get("nn_act", "relu")),
            solver=str(args.get("nn_sol", "adam")),
            alpha=float(args.get("nn_alpha", 0.0001)),
            learning_rate_init=float(args.get("nn_lr", 0.001)),
            max_iter=int(args.get("nn_iter", 200)),
            random_state=42
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # --- 6. EcoRETINA ---
    elif algo == 'EcoRETINA':
        if not ECO_RETINA_AVAILABLE:
            raise ImportError("EcoRETINA V3 non disponible")
    
        raw_params = str(args.get('eco_params', '[-1.0, 0.0, 1.0]')).strip('[]')
        eco_params_list = [float(x.strip()) for x in raw_params.split(',') if x.strip()]
    
        cont_names = list(args.get('cont_names', []))
        dummy_names = list(args.get('dummy_names', []))
        
        cont_set = set(cont_names)
        dummy_set = set(dummy_names)
        
        con_cols_indices = [int(i) for i, f in enumerate(features) if f in cont_set]
        dummy_cols_indices = [int(i) for i, f in enumerate(features) if f in dummy_set]
        
        if not con_cols_indices and not dummy_cols_indices:
            con_cols_indices = list(range(len(features)))
            dummy_cols_indices = []
    
        cross_dummy_bool = True if str(args.get('eco_cross_dummy', 'False')).lower() in ['true', '1'] else False
        add_log_bool = True if str(args.get('eco_add_log', 'False')).lower() in ['true', '1'] else False
        add_relu_bool = True if str(args.get('eco_add_relu', 'False')).lower() in ['true', '1'] else False
    
        model = EcoRETINA()
        model.fit(
            y=y_train, 
            X=X_train, 
            con_cols_indices=con_cols_indices,
            dummy_cols_indices=dummy_cols_indices,
            col_names=features, 
            params=eco_params_list,
            loss=args.get('eco_loss', 'mse'), 
            grid=float(args.get('eco_grid', 0.005)), 
            reg_type=args.get('eco_reg_type', 'linear'), 
            cross_dummy=cross_dummy_bool,
            max_r2=float(args.get('eco_max_r2', 0.99)),
            max_instances=int(args.get('eco_max_instances', 10000)),
            max_reg=int(args.get('eco_max_reg', 30)),
            chunk_size=int(args.get('eco_chunk_size', 50)),
            model_step=int(args.get('eco_model_step', 1)),
            seed=int(args.get('eco_seed', 8)), 
            cov_type=args.get('eco_cov_type', 'nonrobust'),
            handle_zeros=args.get('eco_handle_zeros', 'prevent_division'),
            epsilon=args.get('eco_epsilon', 'auto'),
            add_log=add_log_bool,
            add_relu=add_relu_bool
        )
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if hasattr(model, 'sm_model') and hasattr(model.sm_model, 'params'):
            feature_names_final = list(model.sm_model.params.index)
        elif hasattr(model, 'feature_names_'):
            feature_names_final = list(model.feature_names_)
        elif hasattr(model, 'selected_features_'):
            feature_names_final = list(model.selected_features_)

    gc.collect()

    return {
        'model': model, 
        'model_name': algo, 
        'target_col': target, 
        'raw_features': features, 
        'feature_names': feature_names_final,
        'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else y_test, 
        'y_test_pred': y_test_pred.tolist() if hasattr(y_test_pred, 'tolist') else y_test_pred,
        'metrics': {
            'r2_tr': float(r2_score(y_train, y_train_pred)),
            'mape_tr': float(mean_absolute_percentage_error(y_train, y_train_pred) * 100),
            'r2_te': float(r2_score(y_test, y_test_pred)),
            'rmse_te': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'mape_te': float(mean_absolute_percentage_error(y_test, y_test_pred) * 100),
            'emissions': emissions
        }
    }


    if tracker is not None:
            try:
                emissions = tracker.stop()
                if emissions is None:
                    emissions = 0.0
            except Exception as e:
                print(f"[CODECARBON WARNING] Tracker stop error: {e}")
                emissions = 0.0



# ==========================================
# 4. INTERFACE UTILISATEUR & GESTION MULTI-CLIENTS
# ==========================================


def on_row_right_click(event, state):
    args = getattr(event, 'args', [])
    run_id = None
    
    if isinstance(args, list) and len(args) >= 2 and isinstance(args[1], dict):
        run_id = args[1].get('run')
    elif isinstance(args, dict):
        run_id = args.get('run')
        
    if run_id:
        clean_id = str(run_id).replace('⭐', '').replace('★', '').strip()
        state.selected_context_run_id = clean_id


@ui.page('/')
def main_page():
    # État isolé unique pour CET utilisateur connecté
    state = Workspace()
    
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

    async def trigger_pipeline_execution():
                try:
                    if state.df is None: 
                        return ui.notify("No active dataset loaded!", type='warning')
                        
                    target = state.main_target_select.value
                    algo = state.algo_choice.value
                    
                    chosen_cont = [col for col, cb in state.cont_checkboxes.items() if cb.value]
                    chosen_dummy = [col for col, cb in state.dummy_checkboxes.items() if cb.value]
                    
                    cont_features = [v for v in chosen_cont if v != target]
                    dummy_features = [v for v in chosen_dummy if v != target]
                    
                    if not cont_features and not dummy_features: 
                        return ui.notify("Please select at least one explanatory feature (Continuous or Dummy)!", type='warning')
        
                    state.algo_status_lbl.text = "Running pipeline calculations..."
                    state.btn_run.disable()
                    state.btn_stop.enable()
                    state.algo_status_lbl.update()

                    chosen_cont = [col for col, cb in state.cont_checkboxes.items() if cb.value]
                    chosen_dummy = [col for col, cb in state.dummy_checkboxes.items() if cb.value]
                    
                    config_args = {
                        "algo": algo,
                        "target_col": target,
                        'cont_names': [c for c in chosen_cont if c != target],
                        'dummy_names': [d for d in chosen_dummy if d != target],
                        'eco_loss': str(getattr(state, 'eco_loss', type('obj', (), {'value': "mse"})).value),
                        'eco_reg_type': str(getattr(state, 'eco_reg_type', type('obj', (), {'value': "linear"})).value),
                        'eco_cross_dummy': str(getattr(state, 'eco_cross_dummy', type('obj', (), {'value': "False"})).value),
                        'eco_cov_type': str(getattr(state, 'eco_cov_type', type('obj', (), {'value': "nonrobust"})).value),
                        'eco_grid': float(getattr(state, 'eco_grid', type('obj', (), {'value': 0.005})).value),
                        'eco_max_reg': int(getattr(state, 'eco_max_reg', type('obj', (), {'value': 100})).value),
                        'eco_chunk_size': int(getattr(state, 'eco_chunk_size', type('obj', (), {'value': 500})).value),
                        'eco_seed': int(getattr(state, 'eco_seed', type('obj', (), {'value': 8})).value),
                        'eco_params': str(getattr(state, 'eco_params', type('obj', (), {'value': '[-1.0, 0.0, 1.0]'})).value),
                        'eco_max_r2': float(getattr(state, 'eco_max_r2', type('obj', (), {'value': 0.99})).value),
                        'eco_max_instances': int(getattr(state, 'eco_max_instances', type('obj', (), {'value': 100000})).value),
                        'eco_model_step': int(getattr(state, 'eco_model_step', type('obj', (), {'value': 1})).value),
                        'eco_handle_zeros': str(getattr(state, 'eco_handle_zeros', type('obj', (), {'value': 'prevent_division'})).value),
                        'eco_epsilon': str(getattr(state, 'eco_epsilon', type('obj', (), {'value': 'auto'})).value),
                        'eco_add_log': str(getattr(state, 'eco_add_log', type('obj', (), {'value': 'False'})).value),
                        'eco_add_relu': str(getattr(state, 'eco_add_relu', type('obj', (), {'value': 'False'})).value),
                        
                        'fit_intercept': str(getattr(state, 'ols_fit_intercept', getattr(state, 'fit_intercept_input', type('obj', (), {'value': 'True'}))).value),
                        'alpha': float(getattr(state, 'alpha_input', type('obj', (), {'value': 0.01})).value),
                        'max_iter': int(getattr(state, 'max_iter_input', getattr(state, 'nn_max_iter', type('obj', (), {'value': 1000}))).value),
                        'tol': float(getattr(state, 'tol_input', type('obj', (), {'value': 0.0001})).value),
                        'ridge_solver': str(getattr(state, 'ridge_solver', type('obj', (), {'value': 'auto'})).value),
                        'en_l1_ratio': float(getattr(state, 'en_l1_ratio', type('obj', (), {'value': 0.5})).value),
                        
                        'xgb_n': int(getattr(state, 'xgb_n', type('obj', (), {'value': 100})).value),
                        'xgb_depth': int(getattr(state, 'xgb_depth', type('obj', (), {'value': 6})).value),
                        'xgb_lr': float(getattr(state, 'xgb_lr', type('obj', (), {'value': 0.1})).value),
                        'xgb_subsample': float(getattr(state, 'xgb_subsample', type('obj', (), {'value': 1.0})).value),
                        'xgb_colsample': float(getattr(state, 'xgb_colsample', type('obj', (), {'value': 1.0})).value),
                        'xgb_gamma': float(getattr(state, 'xgb_gamma', type('obj', (), {'value': 0.0})).value),
                        'xgb_alpha': float(getattr(state, 'xgb_alpha', type('obj', (), {'value': 0.0})).value),
                        'xgb_lambda': float(getattr(state, 'xgb_lambda', type('obj', (), {'value': 1.0})).value),
                        
                        'rf_n_estimators': int(getattr(state, 'rf_n_estimators', type('obj', (), {'value': 100})).value),
                        'rf_max_depth': int(getattr(state, 'rf_max_depth', type('obj', (), {'value': 0})).value),
                        'rf_split': int(getattr(state, 'rf_min_split', type('obj', (), {'value': 2})).value),
                        'rf_leaf': int(getattr(state, 'rf_min_leaf', type('obj', (), {'value': 1})).value),
                        'rf_max_features': str(getattr(state, 'rf_max_features', type('obj', (), {'value': '1.0'})).value),
                        
                        'nn_layers': str(getattr(state, 'nn_layers', type('obj', (), {'value': '100,50'})).value),
                        'nn_act': str(getattr(state, 'nn_act', type('obj', (), {'value': 'relu'})).value),
                        'nn_sol': str(getattr(state, 'nn_sol', type('obj', (), {'value': 'adam'})).value),
                        'nn_alpha': float(getattr(state, 'nn_alpha', type('obj', (), {'value': 0.0001})).value),
                        'nn_lr': float(getattr(state, 'nn_lr', type('obj', (), {'value': 0.001})).value),
                        'nn_iter': int(getattr(state, 'nn_iter', type('obj', (), {'value': 200})).value),
                    }
        
                    flat_selected_features = cont_features + dummy_features
        
                    loop = asyncio.get_event_loop()
                    res = await loop.run_in_executor(None, execute_ml_math_core, state.df, algo, target, flat_selected_features, config_args)
                
                    run_id = f"Run_{time.strftime('%H%M%S')}"
                    state.run_history[run_id] = res
                    state.latest_run_by_algo[algo] = run_id
                    
                    # Évite d'accumuler plus de 5 runs en RAM
                    if len(state.run_history) > 5:
                        oldest = next(iter(state.run_history))
                        del state.run_history[oldest]
                    
                    co2_val = res['metrics'].get('emissions', np.nan)
                    if co2_val is not None and not np.isnan(co2_val):
                        co2_str = f"{co2_val:.6f}" if co2_val >= 1e-5 else f"{co2_val:.2e}"
                    else:
                        co2_str = "N/A"

                    state.compare_table_ui.add_rows([{
                        'run': run_id, 
                        'algo': algo,
                        'r2_tr': f"{res['metrics']['r2_tr']:.4f}", 
                        'mape_tr': f"{res['metrics']['mape_tr']:.2f}%",
                        'r2_te': f"{res['metrics']['r2_te']:.4f}", 
                        'rmse_te': f"{res['metrics']['rmse_te']:.4f}",
                        'mape_te': f"{res['metrics']['mape_te']:.2f}%", 
                        'co2': co2_str
                    }])
                    
                    state.algo_status_lbl.text = f"Model saved: {run_id}"
                    state.log(f"Pipeline completed successfully for {algo}.")
                    
                except Exception as ex:
                    ui.notify(f"Pipeline initiation failed: {str(ex)}", type='negative')
                    print(f"[CRITICAL ERROR] trigger_pipeline_execution crashed: {str(ex)}")
                    
                finally:
                    import gc
                    gc.collect()
                    state.btn_run.enable()
                    state.btn_stop.disable()

    # --- MÉTHODES UI LOCALES (SCOPÉES SUR state) ---
    def sync_all_comboboxes():
        if state.df is None: return
        cols = [str(c) for c in state.df.columns]
        num_cols = [str(c) for c in state.df.select_dtypes(include=[np.number]).columns]
        
        state.outlier_select.options = num_cols
        if num_cols: state.outlier_select.value = num_cols[0]
        state.outlier_select.update()
        
        state.main_target_select.options = cols
        if cols: state.main_target_select.value = cols[0]
        state.main_target_select.update()
        
        target = state.main_target_select.value
        suggested_cont = [c for c in num_cols if c != target and state.df[c].nunique() > 2]
        suggested_dummy = [c for c in cols if c != target and (c not in suggested_cont)]
        
        state.cont_scroll_area.clear()
        state.cont_checkboxes = {}
        with state.cont_scroll_area:
            for col in cols:
                is_checked = col in suggested_cont
                state.cont_checkboxes[col] = ui.checkbox(col, value=is_checked).classes('text-xs text-slate-300 block')
                
        state.dummy_scroll_area.clear()
        state.dummy_checkboxes = {}
        with state.dummy_scroll_area:
            for col in cols:
                is_checked = col in suggested_dummy
                state.dummy_checkboxes[col] = ui.checkbox(col, value=is_checked).classes('text-xs text-slate-300 block')

    async def import_main_dataset_from_event(e):
        try:
            file_obj = e.file
            raw_bytes = await file_obj.read()
            df_raw = await run.io_bound(_parse_csv_bytes, raw_bytes)
            
            # Conversion automatique des colonnes numériques en float32 dès l'import
            num_cols = df_raw.select_dtypes(include=[np.number]).columns
            df_raw[num_cols] = df_raw[num_cols].astype(np.float32)
            
            state.df = df_raw
            state.save_state("Import dataset")
            state.log(f"Dataset successfully loaded ({len(state.df)} rows, {len(state.df.columns)} variables).")
            sync_all_comboboxes()
        except Exception as ex:
            state.log(f"Dataset import failed: {str(ex)}")
            print(traceback.format_exc())

    async def import_predict_dataset_from_event(e):
        try:
            file_obj = e.file
            raw_bytes = await file_obj.read()
            state.df_predict = await run.io_bound(_parse_csv_bytes, raw_bytes)
            state.predict_file_lbl.text = f"Inference dataset validated ({len(state.df_predict)} rows)"
            state.log("Inference dataset loaded successfully.")
        except Exception as ex:
            ui.notify(f"Import error: {str(ex)}", type='negative')
            state.log(f"Failed to load inference dataset: {str(ex)}")

    def toggle_split_view(strategy):
        state.split_strategy = strategy
        if strategy == 'Train/Test Split':
            state.split_container_ui.remove_class('hidden')
            state.kfold_container_ui.add_class('hidden')
        else:
            state.split_container_ui.add_class('hidden')
            state.kfold_container_ui.remove_class('hidden')

    def on_outlier_variable_select(e):
        if state.df is None or not e.value: 
            return
            
        col = e.value
        try:
            stats_desc = state.df[col].describe()
            stats_html = (
                f"<div class='font-mono text-xs text-emerald-400 bg-black/40 p-3 rounded-xl border border-slate-800/80 leading-relaxed'>"
                f"<b>📊 DESCRIPTIVE STATISTICS FOR '{col.upper()}'</b><br>"
                f"--------------------------------------------------<br>"
                f"  Count    : {int(stats_desc.get('count', 0)):<12} |    Min      : {stats_desc.get('min', 0):.4f}<br>"
                f"  Mean     : {stats_desc.get('mean', 0):<12.4f} |    25% (Q1) : {stats_desc.get('25%', 0):.4f}<br>"
                f"  Std Dev  : {stats_desc.get('std', 0):<12.4f} |    50% (Med): {stats_desc.get('50%', 0):.4f}<br>"
                f"  Variance : {state.df[col].var():<12.4f} |    75% (Q3) : {stats_desc.get('75%', 0):.4f}<br>"
                f"  Skewness : {state.df[col].skew():<12.4f} |    Max      : {stats_desc.get('max', 0):.4f}"
                f"</div>"
            )
            state.outlier_stats_area.content = stats_html
            state.outlier_stats_area.update()
            
            state.outlier_min.placeholder = f"Min: {stats_desc.get('min', 0):.2f}"
            state.outlier_max.placeholder = f"Max: {stats_desc.get('max', 0):.2f}"
            state.outlier_min.update()
            state.outlier_max.update()
        except Exception as ex:
            state.outlier_stats_area.content = f"<div class='text-red-400 font-mono text-xs'>Error computing statistics: {str(ex)}</div>"
            state.outlier_stats_area.update()

    def process_outliers():
        if state.df is None or not state.outlier_select.value: 
            return
            
        col = state.outlier_select.value
        raw_min = state.outlier_min.value
        raw_max = state.outlier_max.value
        
        if raw_min is None and raw_max is None: 
            return ui.notify("Please specify at least a Lower or an Upper boundary!", type='warning')
        
        state.save_state(f"Outliers on {col}")
        action_type = state.outlier_action.value
        
        if 'Winsorize' in action_type:
            mn = float(raw_min) if raw_min is not None else float(state.df[col].min())
            mx = float(raw_max) if raw_max is not None else float(state.df[col].max())
            state.df[col] = state.df[col].clip(lower=mn, upper=mx)
            state.log(f"Outlier filter (Winsorize) applied on '{col}' [{mn}, {mx}].")
        else:
            initial_len = len(state.df)
            if raw_min is not None:
                state.df = state.df[state.df[col] >= float(raw_min)]
            if raw_max is not None:
                state.df = state.df[state.df[col] <= float(raw_max)]
                
            dropped_rows = initial_len - len(state.df)
            state.log(f"Outlier filter (Truncate) applied on '{col}'. {dropped_rows} rows removed.")
            
        sync_all_comboboxes()

    def view_main_data():
        if state.df is None: return ui.notify("No dataset loaded!", type='warning')
        
        with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-5xl h-5/6 bg-slate-900 rounded-2xl text-white p-4'):
            ui.label('Dataset Preview, Descriptive Statistics & Distributions').classes('text-md font-bold text-emerald-400')
            
            with ui.tabs().classes('w-full') as view_tabs:
                t_prev = ui.tab('t_preview', label='Preview (Top 50)')
                t_stats = ui.tab('t_stats', label='Descriptive Statistics')
                t_charts = ui.tab('t_charts', label='Distribution Plots')
                
            with ui.tab_panels(view_tabs, value='t_preview').classes('w-full h-full bg-transparent'):
                with ui.tab_panel('t_preview'):
                    clean_rows = [{str(k): str(v) for k, v in record.items()} for record in state.df.head(50).to_dict('records')]
                    ui.table(
                        columns=[{'name': str(c), 'label': str(c), 'field': str(c)} for c in state.df.columns],
                        rows=clean_rows
                    ).classes('w-full bg-slate-950 rounded-xl overflow-hidden')
                    
                with ui.tab_panel('t_stats'):
                    num_df = state.df.select_dtypes(include=[np.number])
                    if not num_df.empty:
                        stats_df = num_df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(4)
                        stats_df.reset_index(inplace=True)
                        stats_df.rename(columns={'index': 'Variable', '25%': 'Q1 (25%)', '50%': 'Median', '75%': 'Q3 (75%)'}, inplace=True)
                        stats_rows = stats_df.to_dict('records')
                        ui.table(
                            columns=[{'name': str(c), 'label': str(c), 'field': str(c)} for c in stats_df.columns],
                            rows=stats_rows
                        ).classes('w-full bg-slate-950 rounded-xl overflow-hidden')
                    else:
                        ui.label('No numeric variables found for descriptive statistics.').classes('text-slate-400')
                        
                with ui.tab_panel('t_charts'):
                    num_df = state.df.select_dtypes(include=[np.number])
                    if not num_df.empty:
                        num_vars = min(6, len(num_df.columns))
                        fig, axes = plt.subplots(2, 3, figsize=(10, 4), facecolor='#0f172a')
                        axes = axes.flatten()
                        for i, col in enumerate(num_df.columns[:num_vars]):
                            axes[i].hist(num_df[col].dropna(), bins=20, color='#38bdf8', edgecolor='white', alpha=0.8)
                            axes[i].set_title(col, color='white', fontsize=9)
                            axes[i].set_facecolor('#1e293b')
                            axes[i].tick_params(colors='white', labelsize=7)
                        for j in range(i + 1, 6):
                            fig.delaxes(axes[j])
                        fig.tight_layout()
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                        buf.seek(0)
                        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                        plt.close(fig)
                        ui.html(f'<div class="flex justify-center"><img src="data:image/png;base64,{img_b64}"/></div>')

            ui.button('Close', on_click=dialog.close).classes('bg-slate-800 rounded-xl self-end mt-2')
        dialog.open()

    def apply_undo():
        if state.history:
            act, prev = state.history.pop()
            state.future.append((act, state.df.copy(deep=True)))
            state.df = prev.copy(deep=True)
            state.log(f"[UNDO] Reverted: {act}")
            sync_all_comboboxes()

    def apply_redo():
        if state.future:
            act, nxt = state.future.pop()
            state.history.append((act, state.df.copy(deep=True)))
            state.df = nxt.copy(deep=True)
            state.log(f"[REDO] Restored: {act}")
            sync_all_comboboxes()

    def refresh_algo_param_view():
        algo = state.algo_choice.value
        state.param_options_frame.clear()
        
        with state.param_options_frame:
            if algo == 'EcoRETINA':
                # Ligne 1
                state.eco_loss = ui.select(['mse', 'mae', 'MAPE', 'AIC', 'BIC'], value='mse', label='Loss').classes('w-28 rounded-xl')
                state.eco_reg_type = ui.select(['linear', 'logit', 'probit'], value='linear', label='Reg Type').classes('w-28 rounded-xl')
                state.eco_cross_dummy = ui.select(['False', 'True'], value='False', label='Cross Dummy').classes('w-28 rounded-xl')
                state.eco_cov_type = ui.select(['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3'], value='nonrobust', label='Cov Type').classes('w-28 rounded-xl')
                
                # Ligne 2
                state.eco_params = ui.input(label='Params (list)', value='[-1.0, 0.0, 1.0]').classes('w-32 rounded-xl')
                state.eco_max_r2 = ui.number(label='Max R²', value=0.99, format='%.2f').classes('w-24 rounded-xl')
                state.eco_grid = ui.number(label='Grid Step', value=0.005, format='%.4f').classes('w-24 rounded-xl')
                state.eco_seed = ui.number(label='Seed', value=8).classes('w-20 rounded-xl')
                
                # Ligne 3
                state.eco_max_instances = ui.number(label='Max Inst.', value=100000).classes('w-28 rounded-xl')
                state.eco_max_reg = ui.number(label='Max Reg', value=100).classes('w-24 rounded-xl')
                state.eco_chunk_size = ui.number(label='Chunk Size', value=500).classes('w-24 rounded-xl')
                state.eco_model_step = ui.number(label='Model Step', value=1).classes('w-24 rounded-xl')
                
                # Ligne 4 (Gestion des zéros et non-linéarités)
                state.eco_handle_zeros = ui.select(['prevent_division', 'translate', 'drop_rows'], value='prevent_division', label='Handle Zeros').classes('w-36 rounded-xl')
                state.eco_epsilon = ui.input(label='Epsilon (Shift)', value='auto').classes('w-28 rounded-xl')
                state.eco_add_log = ui.select(['False', 'True'], value='False', label='Add Logs (ln)').classes('w-28 rounded-xl')
                state.eco_add_relu = ui.select(['False', 'True'], value='False', label='Add ReLU').classes('w-28 rounded-xl')
                
            elif algo in ['OLS', 'Lasso', 'Ridge', 'ElasticNet']:
                state.ols_fit_intercept = ui.select(['True', 'False'], value='True', label='Fit Intercept').classes('w-32 rounded-xl')
                if algo != 'OLS':
                    state.alpha_input = ui.number(label='Alpha (Penalty)', value=0.01, format='%.4f').classes('w-28 rounded-xl')
                if algo in ['Lasso', 'ElasticNet', 'Ridge']:
                    state.max_iter_input = ui.number(label='Max Iterations', value=1000).classes('w-28 rounded-xl')
                    state.tol_input = ui.number(label='Tolerance', value=0.0001, format='%.5f').classes('w-28 rounded-xl')
                if algo == 'Ridge':
                    state.ridge_solver = ui.select(['auto', 'svd', 'cholesky', 'lsqr', 'sag'], value='auto', label='Solver').classes('w-32 rounded-xl')
                if algo == 'ElasticNet':
                    state.en_l1_ratio = ui.number(label='L1 Ratio', value=0.5, format='%.2f').classes('w-24 rounded-xl')
    
            elif algo == 'XGBoost':
                state.xgb_n = ui.number(label='Estimators', value=100).classes('w-24 rounded-xl')
                state.xgb_depth = ui.number(label='Max Depth', value=6).classes('w-24 rounded-xl')
                state.xgb_lr = ui.number(label='Learning Rate', value=0.1, format='%.2f').classes('w-24 rounded-xl')
                state.xgb_subsample = ui.number(label='Subsample', value=1.0, format='%.2f').classes('w-24 rounded-xl')
                state.xgb_colsample = ui.number(label='Colsample', value=1.0, format='%.2f').classes('w-24 rounded-xl')
                state.xgb_gamma = ui.number(label='Gamma', value=0.0, format='%.2f').classes('w-24 rounded-xl')
                state.xgb_alpha = ui.number(label='Alpha (L1)', value=0.0, format='%.2f').classes('w-24 rounded-xl')
                state.xgb_lambda = ui.number(label='Lambda (L2)', value=1.0, format='%.2f').classes('w-24 rounded-xl')
    
            elif algo == 'Random Forest':
                state.rf_n_estimators = ui.number(label='Estimators', value=100).classes('w-24 rounded-xl')
                state.rf_max_depth = ui.number(label='Max Depth (0=None)', value=12).classes('w-28 rounded-xl')
                state.rf_min_split = ui.number(label='Min Split', value=2).classes('w-24 rounded-xl')
                state.rf_min_leaf = ui.number(label='Min Leaf', value=1).classes('w-24 rounded-xl')
                state.rf_max_features = ui.select(['1.0', 'sqrt', 'log2'], value='1.0', label='Max Features').classes('w-28 rounded-xl')
    
            elif algo == 'Neural Network':
                state.nn_layers = ui.input(label='Hidden Layers', value='100, 50').classes('w-32 rounded-xl')
                state.nn_act = ui.select(['relu', 'tanh', 'logistic', 'identity'], value='relu', label='Activation').classes('w-32 rounded-xl')
                state.nn_sol = ui.select(['adam', 'sgd', 'lbfgs'], value='adam', label='Solver').classes('w-32 rounded-xl')
                state.nn_alpha = ui.number(label='Alpha (L2)', value=0.0001, format='%.5f').classes('w-28 rounded-xl')
                state.nn_lr = ui.number(label='Learning Rate', value=0.001, format='%.4f').classes('w-28 rounded-xl')
                state.nn_iter = ui.number(label='Max Iterations', value=200).classes('w-28 rounded-xl')

    def export_comparison_matrix():
        if not state.compare_table_ui.rows: return ui.notify("No comparison data available", type='warning')
        output = io.StringIO()
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
        if state.df_predict is None: return ui.notify("Please load an inference dataset first!", type='warning')
        run_id = state.predict_run_select.value
        if not run_id: return ui.notify("Please select a trained model!", type='warning')
        
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
            ui.notify(f"Predictions generated in column '{pred_col}'!")
        except Exception as ex:
            ui.notify(f"Inference calculation error: {str(ex)}", type='negative')

    def view_predict_data():
        if state.df_predict is None: return ui.notify("No prediction dataset loaded!", type='warning')
        with ui.dialog() as dialog, ui.card().classes('w-11/12 max-w-5xl h-5/6 bg-slate-900 rounded-2xl text-white'):
            ui.label('Predictions Overview').classes('text-md font-bold text-emerald-400')
            ui.table(
                columns=[{'name': c, 'label': c, 'field': c} for c in state.df_predict.columns],
                rows=state.df_predict.head(50).to_dict('records')
            ).classes('w-full bg-slate-950 rounded-xl overflow-hidden')
            ui.button('Close', on_click=dialog.close).classes('bg-slate-800 rounded-xl self-end')
        dialog.open()

    def export_predicted_csv():
        if state.df_predict is None: return
        csv_buf = io.StringIO()
        state.df_predict.to_csv(csv_buf, index=False)
        ui.download(csv_buf.getvalue().encode('utf-8'), 'EcoRETINA_Predictions_Output.csv')

    # --- RENDU VISUEL ET PANNEAUX ---
    with ui.header().classes('bg-slate-900/80 backdrop-blur-md text-white items-center justify-between p-4 shadow-xl m-4 rounded-2xl border border-slate-800'):
        with ui.row().classes('items-center gap-4'):
            ui.label('≡').classes('text-3xl cursor-pointer hover:text-emerald-400 transition-colors').on('click', lambda: left_drawer.toggle())
            ui.label('EcoRETINA Intelligence').classes('text-xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-300')
        with ui.row().classes('items-center gap-3'):
            ui.button('↩ Undo', on_click=apply_undo).props('flat color=white').classes('hover:bg-slate-800 rounded-xl')
            ui.button('↪ Redo', on_click=apply_redo).props('flat color=white').classes('hover:bg-slate-800 rounded-xl')
            ui.button('🤖 AI Copilot', on_click=lambda: right_drawer.toggle()).classes('bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl font-bold shadow-lg shadow-blue-500/20')

    with ui.left_drawer(value=False).classes('bg-slate-900/90 backdrop-blur-md p-4 text-white rounded-r-3xl border-r border-slate-800') as left_drawer:
        ui.label('Navigation').classes('text-sm uppercase tracking-wider font-bold p-2 text-slate-400 border-b border-slate-800 w-full mb-4')
        ui.button('Main Workspace', on_click=lambda: main_tabs.set_value('workspace')).classes('w-full justify-start rounded-xl mb-2 py-3 bg-slate-800/50 hover:bg-slate-800')
        ui.button('Activity Log', on_click=lambda: main_tabs.set_value('logs')).classes('w-full justify-start rounded-xl mb-2 py-3 bg-slate-800/50 hover:bg-slate-800')
        ui.button('Tutorial & Docs', on_click=lambda: main_tabs.set_value('tutorial')).classes('w-full justify-start rounded-xl mb-2 py-3 bg-slate-800/50 hover:bg-slate-800')

    with ui.right_drawer(value=False).classes('bg-slate-900/90 backdrop-blur-md p-4 text-white rounded-l-3xl border-l border-slate-800') as right_drawer:
        ui.label('AI Assistant').classes('text-lg font-black text-emerald-400 mb-2')
        provider_ui = ui.select(["Google Gemini", "OpenAI (ChatGPT)", "Groq", "Claude (Anthropic)"], value="Google Gemini").classes('w-full rounded-xl')
        key_ui = ui.input(placeholder='API Key', password=True).classes('w-full rounded-xl')
        
        chat_container = ui.scroll_area().classes('w-full h-96 bg-slate-950/60 p-3 rounded-2xl border border-slate-800 my-4 shadow-inner')
        
        async def connect_ai():
            if not key_ui.value: return ui.notify("Missing API Key!", type='warning')
            state.ai_agent = EcoRetinaChatAgent(key_ui.value, provider_ui.value)
            state.log(f"AI Agent successfully connected via {provider_ui.value}")
        
        ui.button('Connect AI Agent', on_click=connect_ai).classes('w-full bg-emerald-600 rounded-xl font-bold shadow-lg shadow-emerald-500/20')
        
        async def submit_chat():
            msg = state.chat_input.value.strip()
            if not msg or not state.ai_agent: 
                return
                
            state.chat_input.disable()
            state.chat_send_btn.disable()
            
            with chat_container:
                ui.label(f"User: {msg}").classes('text-blue-400 font-bold block mt-2 text-sm')
                ai_bubble = ui.label("⏳ Thinking...").classes(
                    'text-slate-200 block ml-2 bg-slate-800/80 p-3 rounded-2xl text-sm border border-slate-700/50'
                )
                chat_container.scroll_to(percent=1.0)
                
            state.chat_input.value = ''
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_ai_task, msg, ai_bubble)
            
            state.chat_input.enable()
            state.chat_send_btn.enable()
            chat_container.scroll_to(percent=1.0)

        chat_row = ui.row().classes('w-full mt-4 items-center gap-2 no-wrap')
        with chat_row:
            state.chat_input = ui.input(placeholder='Ask a question...').classes('flex-grow rounded-xl')
            state.chat_send_btn = ui.button('Send', on_click=submit_chat).classes('bg-emerald-600 rounded-xl font-bold px-4')
            
        state.chat_input.on('keydown.enter', submit_chat)

        def run_ai_task(msg, ui_element):
            asyncio.run(state.ai_agent.ask(msg, ui_element))

    with ui.tab_panels(ui.tabs(), value='workspace').classes('w-full bg-transparent px-4') as main_tabs:
        
        with ui.tab_panel('workspace'):
            with ui.tabs().classes('w-full bg-slate-900/40 p-2 rounded-2xl border border-slate-800/60 text-white') as step_tabs:
                t1 = ui.tab('t_data', label='1. Data & Pre-Processing').classes('rounded-xl')
                t2 = ui.tab('t_algo', label='2. Algorithms & Params').classes('rounded-xl')
                t3 = ui.tab('t_compare', label='3. Compare Results').classes('rounded-xl')
                t4 = ui.tab('t_predict', label='4. Predict (New Data)').classes('rounded-xl')

            with ui.tab_panels(step_tabs, value='t_data').classes('w-full bg-transparent pt-4 overflow-visible') as step_panels:
                
                # STEP 1 : DATA
                with ui.tab_panel('t_data'):
                    with ui.row().classes('w-full gap-6'):
                        with ui.card().classes('w-full md:w-[48%] bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                            ui.label('Dataset Import & Sample Strategy').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-2')
                            
                            ui.upload(label='Dataset import', on_upload=import_main_dataset_from_event).classes('w-full rounded-2xl')
                            
                            ui.select(['Train/Test Split', 'K-Fold Cross Validation'], value='Train/Test Split', on_change=lambda e: toggle_split_view(e.value)).classes('w-full mt-4 rounded-xl')
                            with ui.column().classes('w-full') as split_container:
                                state.split_slider = ui.slider(min=50, max=100, value=80).classes('w-full mt-2')
                                ui.label().bind_text_from(state.split_slider, 'value', backward=lambda v: f"Train Ratio: {v}%")
                            with ui.column().classes('w-full hidden') as kfold_container:
                                state.kfold_input = ui.number(label='Number of Folds (K)', value=5, min=2).classes('w-full rounded-xl')
                            state.split_container_ui = split_container
                            state.kfold_container_ui = kfold_container

                            ui.button('Visualize Dataset & Statistics', on_click=view_main_data).classes('bg-blue-600/90 w-full mt-6 rounded-xl py-2 font-bold')

                        with ui.card().classes('w-full md:w-[48%] bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                            ui.label('Advanced Outlier Management').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-2')
                            
                            with ui.expansion('Outlier Filtering Options', icon='analytics').classes('w-full bg-slate-950/50 border border-slate-800 rounded-xl mb-3'):
                                state.outlier_select = ui.select([], label='Select Numeric Variable', on_change=on_outlier_variable_select).classes('w-full')
                                state.outlier_stats_area = ui.html('<div class="text-slate-400 font-mono text-xs p-2 bg-slate-950 rounded-xl border border-slate-800/50">Select a variable to view descriptive statistics...</div>').classes('w-full my-2')
                                with ui.row().classes('w-full gap-2'):
                                    state.outlier_min = ui.number(label='Lower Bound').classes('w-[47%]')
                                    state.outlier_max = ui.number(label='Upper Bound').classes('w-[47%]')
                                
                                state.outlier_action = ui.select(
                                    ['Winsorize (Cap extreme values at lower/upper thresholds)', 'Truncate (Remove rows falling outside thresholds)'], 
                                    value='Winsorize (Cap extreme values at lower/upper thresholds)'
                                ).classes('w-full')
                                
                                ui.label('Note: Winsorization caps outliers at fixed thresholds without reducing sample size (N).').classes('text-xs text-slate-400 italic my-1')
                                ui.button('Apply Outlier Filter', on_click=process_outliers).classes('w-full bg-amber-600 rounded-xl mt-2')

                # STEP 2 : ALGORITHMS
                with ui.tab_panel('t_algo'):
                    with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                        ui.label('Algorithm Configuration').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-4')
                        
                        with ui.row().classes('w-full gap-4 items-center'):
                            state.algo_choice = ui.select(
                                ['EcoRETINA', 'OLS', 'Lasso', 'Ridge', 'ElasticNet', 'XGBoost', 'Random Forest', 'Neural Network'], 
                                value='EcoRETINA', 
                                on_change=refresh_algo_param_view
                            ).classes('w-1/3 rounded-xl')
                            
                            state.main_target_select = ui.select([], label='Target Variable (Y)').classes('w-1/3 rounded-xl')
                        
                        state.param_options_frame = ui.row().classes('w-full bg-slate-950/50 p-4 rounded-xl border border-slate-800 mt-4')
                        
                        ui.label('Feature Selection (Predictors X)').classes('text-md uppercase tracking-wider font-bold text-slate-400 mt-6 mb-2')
                        
                        with ui.row().classes('w-full gap-4 mt-2 no-wrap'):
                            with ui.column().classes('w-1/2'):
                                ui.label('Continuous Features (X)').classes('text-xs uppercase font-bold text-slate-400 mb-1')
                                with ui.expansion('Select Continuous Variables', icon='analytics').classes('w-full rounded-xl border border-slate-800 bg-slate-950'):
                                    state.cont_scroll_area = ui.scroll_area().classes('h-60 p-2')
                                    with state.cont_scroll_area:
                                        state.cont_checkboxes = {}

                            with ui.column().classes('w-1/2'):
                                ui.label('Dummy Variables (X)').classes('text-xs uppercase font-bold text-slate-400 mb-1')
                                with ui.expansion('Select Dummy Variables', icon='tune').classes('w-full rounded-xl border border-slate-800 bg-slate-950'):
                                    state.dummy_scroll_area = ui.scroll_area().classes('h-60 p-2')
                                    with state.dummy_scroll_area:
                                        state.dummy_checkboxes = {}

                        ui.label('Execution Controls').classes('text-md uppercase tracking-wider font-bold text-slate-400 mt-6 mb-2')
                        
                        with ui.row().classes('w-full items-center bg-slate-950/40 p-4 rounded-xl border border-slate-800/60 gap-4 mt-2'):
                            state.btn_run = ui.button(
                                '► Run Model Pipeline', 
                                on_click=lambda: trigger_pipeline_execution()
                            ).classes('bg-emerald-600 hover:bg-emerald-700 text-white font-black text-md px-6 py-2 rounded-xl shadow-lg shadow-emerald-500/10 transition-all')
                            
                            state.algo_status_lbl = ui.label('System Ready. Waiting for execution...').classes('text-sm text-slate-400 font-mono flex-grow italic')
                            
                            state.btn_stop = ui.button(
                                'Stop', 
                                on_click=lambda: ui.notify('Aborting calculations...', type='warning')
                            ).classes('bg-red-600/20 hover:bg-red-600 text-red-400 hover:text-white rounded-xl font-bold px-4 transition-all')
                            
                            state.btn_stop.disable()

                # STEP 3 : COMPARE
                
                with ui.tab_panel('t_compare'):
                    with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                        ui.label('Global Comparative Benchmark').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-1')

                        state.compare_table_ui = ui.table(
                            columns=[
                                {'name': 'run', 'label': 'Run ID', 'field': 'run', 'align': 'center'},
                                {'name': 'algo', 'label': 'Algorithm', 'field': 'algo', 'align': 'center'},
                                {'name': 'r2_tr', 'label': 'R² Train', 'field': 'r2_tr', 'align': 'center'},
                                {'name': 'mape_tr', 'label': 'MAPE Train', 'field': 'mape_tr', 'align': 'center'},
                                {'name': 'r2_te', 'label': 'R² Test', 'field': 'r2_te', 'align': 'center'},
                                {'name': 'rmse_te', 'label': 'RMSE Test', 'field': 'rmse_te', 'align': 'center'},
                                {'name': 'mape_te', 'label': 'MAPE Test', 'field': 'mape_te', 'align': 'center'},
                                {'name': 'co2', 'label': 'Carbon (kgCO2eq)', 'field': 'co2', 'align': 'center'},
                                {'name': 'action', 'label': 'Report', 'field': 'action', 'align': 'center'},
                            ], rows=[]
                        ).classes('w-full bg-slate-950 text-white rounded-xl overflow-hidden border border-slate-800')

                        state.compare_table_ui.add_slot(
                            'body-cell-action',
                            r'''
                            <q-td :props="props">
                                <q-btn 
                                    dense 
                                    size="sm" 
                                    class="bg-emerald-600 text-white font-bold px-3 py-1 rounded-lg hover:bg-emerald-500"
                                    icon="analytics" 
                                    label="View Report" 
                                    @click="$parent.$emit('open_report', props.row.run)"
                                />
                            </q-td>
                            '''
                        )

                        def on_open_report(e):
                            raw_id = str(e.args) if not isinstance(e.args, str) else e.args
                            clean_id = raw_id.replace('⭐', '').replace('★', '').strip("[]'\" ")
                            if clean_id in state.run_history:
                                show_summary_dialog(clean_id, state)
                            else:
                                ui.notify(f"Run {clean_id} not found.", type='warning')

                        state.compare_table_ui.on('open_report', on_open_report)

                        with ui.row().classes('w-full justify-between mt-6'):
                            ui.button('Clear Table', on_click=lambda: state.compare_table_ui.rows.clear()).classes('bg-red-600/80 rounded-xl')
                            ui.button('Export Comparison CSV', on_click=export_comparison_matrix).classes('bg-emerald-600 rounded-xl font-bold')
                                # STEP 4 : PREDICT
                with ui.tab_panel('t_predict'):
                    with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                        ui.label('Prediction on New Dataset').classes('text-md uppercase tracking-wider font-bold text-emerald-400 mb-4')
                        
                        with ui.row().classes('w-full gap-6'):
                            with ui.card().classes('w-[48%] bg-slate-950/40 p-4 rounded-xl border border-slate-800'):
                                ui.label('1. Load New Dataset').classes('text-sm font-bold text-slate-300 mb-2')
                                ui.upload(label='Browse New Dataset', on_upload=import_predict_dataset_from_event).classes('w-full rounded-xl')
                                state.predict_file_lbl = ui.label('No prediction file loaded').classes('text-slate-400 font-mono text-xs mt-2')
                            
                            with ui.card().classes('w-[48%] bg-slate-950/40 p-4 rounded-xl border border-slate-800'):
                                ui.label('2. Select Trained Model').classes('text-sm font-bold text-slate-300 mb-2')
                                state.predict_run_select = ui.select([], label='Choose a model').classes('w-full rounded-xl')
                                ui.button('Apply Model', on_click=sync_predict_runs).classes('w-full bg-slate-800 rounded-xl text-xs mt-2')

                        with ui.row().classes('w-full justify-between items-center mt-6 border-t border-slate-800 pt-4'):
                            with ui.row().classes('gap-3'):
                                ui.button('Run Prediction', on_click=execute_inference_process).classes('bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl font-bold')
                                ui.button('Visualize Results', on_click=view_predict_data).classes('bg-blue-600 rounded-xl')
                            ui.button('Export Prediction to CSV', on_click=export_predicted_csv).classes('bg-indigo-600 rounded-xl')

        # LOGS & DOCS
        with ui.tab_panel('logs'):
            with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl'):
                ui.label('System Activity Log').classes('text-lg font-bold text-emerald-400 mb-4')
                state.logs_area = ui.html().classes('font-mono text-xs bg-black p-4 rounded-xl w-full h-96 overflow-y-scroll text-green-400 border border-slate-800')
                state.logs_area.content = "<br>".join(state.logs)

        with ui.tab_panel('tutorial'):
            with ui.card().classes('w-full bg-slate-900/60 border border-slate-800 p-6 rounded-2xl shadow-xl text-white'):
                ui.label('Tutorial').classes('text-lg font-bold text-emerald-400 mb-4')
                ui.markdown("# EcoRETINA ML Workbench - User Guide")

    refresh_algo_param_view()
    
    def animate_moving_star():
        if not state.compare_table_ui.rows:
            return
            
        star_icon = "⭐" if state.star_phase % 2 == 0 else "★"
        state.star_phase += 1
        
        best_r2 = -float('inf')
        best_row_idx = -1
        
        for idx, row in enumerate(state.compare_table_ui.rows):
            try:
                r2_val = float(str(row['r2_te']).replace('⭐', '').replace('★', '').strip())
                if r2_val > best_r2:
                    best_r2 = r2_val
                    best_row_idx = idx
            except ValueError:
                pass
                
        if best_row_idx != -1:
            for idx, row in enumerate(state.compare_table_ui.rows):
                clean_r2 = str(row['r2_te']).replace('⭐', '').replace('★', '').strip()
                if idx == best_row_idx:
                    row['r2_te'] = f"{clean_r2} {star_icon}"
                else:
                    row['r2_te'] = clean_r2
            state.compare_table_ui.update()

    # ui.timer(0.7, animate_moving_star)

# ==========================================
# 5. DÉMARRAGE DU SERVEUR SUR RENDER
# ==========================================
ui.run(
    port=int(os.environ.get('PORT', 8080)), 
    title="EcoRETINA Intelligence", 
    reload=False,
    storage_secret="ecoretina_production_secure_secret_key"
)
