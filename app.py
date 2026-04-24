"""
FinSight AI ─ Next-Level Premium Dashboard
Upgrade of the basic app.py → cinematic dark UI, glass morphism,
Plotly pro charts, animated KPIs, smart advisor, forecast with confidence bands.

Run:  streamlit run app.py
"""

from __future__ import annotations
import warnings
from pathlib import Path
from datetime import date
from utils.helpers import advanced_alerts
from utils.helpers import generate_insights

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import APP_TITLE, APP_SUBTITLE, HEALTHY_SAVINGS_RATE
from utils.helpers import (
    load_data, compute_kpis, get_monthly_summary, get_category_summary,
    get_income_source_summary, detect_overspending, savings_opportunity_report,
    fmt_inr, fmt_pct, build_finance_context,
)
from ml_model import train_model, load_pipeline, predict_month_expenses, MODEL_PATH
from chatbot import FinSightAdvisor


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="FinSight AI", page_icon="💎",
                   layout="wide", initial_sidebar_state="expanded")


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
VIOLET  = "#7C3AED"
VIOLET2 = "#A78BFA"
TEAL    = "#0D9488"
ROSE    = "#E11D48"
AMBER   = "#D97706"
INDIGO  = "#4F46E5"
SKY     = "#0EA5E9"
GOLD    = "#F59E0B"
PAL = ["#2DD4BF", "#38BDF8", "#A78BFA", "#F59E0B", "#F472B6", "#60A5FA", "#34D399", "#FB7185", "#818CF8", "#22D3EE", "#C084FC", "#FBBF24"]

st.markdown(f"""
<style>
/* ─── FONTS ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── CSS VARIABLES ──────────────────────────────────────────────────────── */
:root {{
  --bg0:    #02030A;
  --bg1:    #050816;
  --bg2:    #081226;
  --glass:  rgba(255,255,255,0.055);
  --glassH: rgba(255,255,255,0.095);
  --b0:     rgba(255,255,255,0.10);
  --b1:     rgba(14,165,233,0.55);
  --b2:     rgba(167,139,250,0.48);
  --shadow: 0 22px 64px rgba(0,0,0,.42);
  --shadow2: 0 18px 48px rgba(0,0,0,.34), 0 0 28px rgba(14,165,233,.08);
  --v:      {VIOLET};
  --v2:     {VIOLET2};
  --teal:   {TEAL};
  --rose:   {ROSE};
  --amber:  {AMBER};
  --sky:    {SKY};
  --t1:     #F0EFFE;
  --t2:     #9898C0;
  --t3:     #4A4A70;
  --mono:   'JetBrains Mono', monospace;
  --body:   'Outfit', sans-serif;
}}

/* ─── GLOBAL ─────────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {{
  font-family: var(--body) !important;
  background:
    linear-gradient(145deg, #02030A 0%, #050816 44%, #07172E 100%) !important;
  color: var(--t1) !important;
}}

section.main,
main[data-testid="stAppViewContainer"] {{
  padding-top: 0 !important;
  margin-top: 0 !important;
}}

.block-container {{
  padding-top: 1rem !important;
  margin-top: 0 !important;
}}

h1, h2, h3 {{
  margin-top: 0 !important;
}}

/* animated aurora background */
.stApp::before {{
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 85% 55% at 4%   0%, rgba(124,58,237,.20) 0%, transparent 62%),
    radial-gradient(ellipse 72% 48% at 96%  8%, rgba(14,165,233,.13) 0%, transparent 58%),
    radial-gradient(ellipse 56% 35% at 50% 100%, rgba(13,148,136,.10) 0%, transparent 55%);
  animation: aurora 12s ease-in-out infinite alternate;
}}
@keyframes aurora {{
  0%   {{ opacity:.7; transform: scale(1); }}
  100% {{ opacity:1;  transform: scale(1.04); }}
}}

.glass-card {{
  background:
    linear-gradient(145deg, rgba(255,255,255,.088), rgba(255,255,255,.032)) !important;
  border: 1px solid rgba(255,255,255,.12) !important;
  box-shadow: var(--shadow);
  backdrop-filter: blur(22px) saturate(145%);
  -webkit-backdrop-filter: blur(22px) saturate(145%);
}}
.glow-border {{
  position: relative;
  isolation: isolate;
}}
.glow-border::after {{
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1px;
  background: linear-gradient(135deg, rgba(14,165,233,.78), rgba(167,139,250,.52), rgba(13,148,136,.58));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
  opacity: .72;
}}
.hover-lift {{
  transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease, background .24s ease;
}}
.hover-lift:hover {{
  transform: translateY(-4px);
  box-shadow: 0 26px 72px rgba(0,0,0,.50), 0 0 28px rgba(14,165,233,.10);
}}
.metric-value {{
  text-shadow: 0 0 10px color-mix(in srgb, currentColor 32%, transparent);
  animation: metricGlow 4s ease-in-out infinite alternate;
}}
@keyframes metricGlow {{
  from {{ filter: drop-shadow(0 0 1px rgba(255,255,255,.08)); }}
  to   {{ filter: drop-shadow(0 0 5px rgba(255,255,255,.14)); }}
}}

/* ─── SIDEBAR ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
  background:
    linear-gradient(180deg, rgba(3,5,16,.96) 0%, rgba(5,10,28,.96) 100%) !important;
  border-right: 1px solid rgba(14,165,233,.14) !important;
  box-shadow: 18px 0 70px rgba(0,0,0,.34), inset -1px 0 0 rgba(167,139,250,.08);
  backdrop-filter: blur(20px);
}}

.logo-wrap {{
  padding: 32px 22px 26px;
  border-bottom: 1px solid rgba(255,255,255,.08);
  margin-bottom: 16px;
  text-align: center;
}}
.logo-wrap .brand {{
  font-size: 30px;
  font-weight: 900;
  letter-spacing: 1.8px;
  line-height: 1;
  text-transform: uppercase;
  background: linear-gradient(135deg, #E0F2FE 0%, #38BDF8 36%, #A78BFA 72%, #F8FAFC 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 14px rgba(56,189,248,.28)) drop-shadow(0 0 20px rgba(167,139,250,.14));
}}
.logo-wrap .sub {{
  font-size: 10px; font-weight: 600; letter-spacing: 2.8px;
  text-transform: uppercase; color: var(--t2); margin-top: 8px;
}}
.live-dot {{
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
  color: var(--teal); margin-top: 10px;
}}
.live-dot::before {{
  content: '';
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--teal);
  box-shadow: 0 0 6px var(--teal);
  animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.4}} }}

/* nav radio */
div[role="radiogroup"] label {{
  border-radius: 12px !important; padding: 11px 14px !important;
  color: var(--t2) !important; font-size: 14px !important;
  font-weight: 600 !important; transition: all .22s ease !important;
  border: 1px solid transparent !important;
  margin: 4px 0 !important;
}}
div[role="radiogroup"] label:hover {{
  background: rgba(255,255,255,.07) !important;
  border-color: rgba(14,165,233,.22) !important;
  color: var(--t1) !important;
  transform: translateX(4px);
  box-shadow: 0 10px 32px rgba(14,165,233,.08);
}}
div[role="radiogroup"] label:has(input:checked) {{
  background: linear-gradient(135deg, rgba(14,165,233,.16), rgba(124,58,237,.16)) !important;
  border-color: rgba(14,165,233,.42) !important;
  color: #F8FBFF !important;
  box-shadow: inset 3px 0 0 {SKY}, 0 0 26px rgba(14,165,233,.18);
}}

/* ─── PAGE HEADER ────────────────────────────────────────────────────────── */
.ph {{ padding: 36px 0 28px; border-bottom: 1px solid var(--b0); margin-bottom: 32px; }}
.ph-eye  {{ font-size:11px; font-weight:600; letter-spacing:3px; text-transform:uppercase; color:var(--v2); margin-bottom:6px; }}
.ph-h1   {{ font-size:34px; font-weight:900; letter-spacing:-1.2px; color:var(--t1); margin:0 0 8px; line-height:1.05; }}
.ph-sub  {{ font-size:15px; font-weight:300; color:var(--t2); }}

/* ─── SECTION TITLE ──────────────────────────────────────────────────────── */
.stitle {{
  font-size:16px; font-weight:700; color:var(--t1);
  display:flex; align-items:center; gap:10px; margin:0 0 18px;
}}
.stitle::after {{ content:''; flex:1; height:1px; background:var(--b0); }}

/* ─── KPI CARD ───────────────────────────────────────────────────────────── */
.kc {{
  border-radius:20px; padding:24px 22px;
  position:relative; overflow:hidden;
  min-height: 158px;
  box-shadow: 0 20px 56px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.08), inset 0 -18px 38px rgba(0,0,0,.16);
}}
.kc::before {{
  content:''; position:absolute; inset:-38% -20% auto auto; width:168px; height:168px;
  background: radial-gradient(circle, var(--card-glow, rgba(14,165,233,.16)), transparent 70%);
  filter: blur(10px);
  opacity:.65;
}}
.kc-glow  {{ position:absolute; top:-38px; right:-30px; width:122px; height:122px; border-radius:50%; filter:blur(42px); opacity:.28; background:var(--card-gradient) !important; }}
.kc-icon  {{ font-size:24px; margin-bottom:16px; display:block; filter: drop-shadow(0 0 8px rgba(255,255,255,.12)); }}
.kc-lbl   {{ font-size:10px; font-weight:800; letter-spacing:2.6px; text-transform:uppercase; color:var(--t2); margin-bottom:8px; }}
.kc-val   {{ font-size:33px; font-weight:900; letter-spacing:-.8px; line-height:1; margin-bottom:12px; font-family:var(--body); }}
.kc-val.pos {{ color:{TEAL}; }}
.kc-val.neg {{ color:{ROSE}; }}
.kc-val.neu {{ color:var(--t1); }}
.kc.total-income {{
  --card-gradient: linear-gradient(135deg, #14B8A6, #34D399);
  --card-glow: rgba(45,212,191,.18);
}}
.kc.total-income .kc-val {{ color:#5EEAD4; }}
.kc.total-expenses {{
  --card-gradient: linear-gradient(135deg, #E11D48, #F59E0B);
  --card-glow: rgba(244,63,94,.17);
}}
.kc.total-expenses .kc-val {{ color:#FB7185; }}
.kc.net-savings {{
  --card-gradient: linear-gradient(135deg, #38BDF8, #2563EB);
  --card-glow: rgba(56,189,248,.17);
}}
.kc.net-savings .kc-val {{ color:#7DD3FC; }}
.kc.savings-rate {{
  --card-gradient: linear-gradient(135deg, #A78BFA, #4F46E5);
  --card-glow: rgba(167,139,250,.17);
}}
.kc.savings-rate .kc-val {{ color:#C4B5FD; }}
.badge {{
  display:inline-flex; align-items:center; gap:4px;
  font-size:11px; font-weight:600; padding:3px 10px; border-radius:99px;
}}
.badge.up   {{ background:rgba(13,148,136,.15); color:{TEAL};  }}
.badge.down {{ background:rgba(225,29,72,.15);  color:{ROSE};  }}
.badge.neu  {{ background:rgba(152,152,192,.12); color:var(--t2); }}

/* ─── GLASS PANEL ────────────────────────────────────────────────────────── */
.gp {{
  border-radius:20px; padding:26px;
  position:relative; overflow:hidden;
}}
.gp::before {{
  content:''; position:absolute; inset:0; pointer-events:none;
  background:
    linear-gradient(180deg, rgba(255,255,255,.08), transparent 26%),
    radial-gradient(ellipse at 88% 8%, rgba(14,165,233,.13), transparent 36%);
}}

div[data-testid="stPlotlyChart"] {{
  background:
    linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.028)) !important;
  border: 1px solid rgba(255,255,255,.11) !important;
  border-radius: 20px !important;
  padding: 20px 20px 12px !important;
  box-shadow: var(--shadow2), inset 0 1px 0 rgba(255,255,255,.08), inset 0 -16px 32px rgba(0,0,0,.12);
  backdrop-filter: blur(20px) saturate(135%);
  -webkit-backdrop-filter: blur(20px) saturate(135%);
  transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease;
}}
div[data-testid="stPlotlyChart"]:hover {{
  transform: translateY(-3px);
  border-color: rgba(14,165,233,.26) !important;
  box-shadow: 0 24px 68px rgba(0,0,0,.46), 0 0 24px rgba(14,165,233,.09);
}}

/* ─── ALERT CARDS ────────────────────────────────────────────────────────── */
.ac {{
  border-radius: 20px;
  padding: 18px 20px 18px 22px;
  margin: 12px 0;
  display: flex;
  align-items: flex-start;
  gap: 16px;
  position: relative;
  overflow: hidden;
  isolation: isolate;
  border: 1px solid rgba(255,255,255,.10) !important;
  box-shadow: 0 20px 54px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.06);
}}
.ac::before {{
  content:'';
  position:absolute;
  inset:0;
  pointer-events:none;
  opacity:.85;
}}
.ac-bar {{
  position: absolute;
  left: 0;
  top: 16px;
  bottom: 16px;
  width: 4px;
  border-radius: 99px;
  background: var(--ac-accent, rgba(255,255,255,.32));
  box-shadow: 0 0 20px var(--ac-accent, rgba(255,255,255,.22));
}}
.ac.hi {{
  --ac-accent: rgba(251,113,133,.96);
  --ac-soft: rgba(225,29,72,.22);
  background: linear-gradient(135deg, rgba(32,10,18,.92), rgba(15,18,32,.94)) !important;
  border-color: rgba(251,113,133,.26) !important;
  box-shadow: 0 22px 58px rgba(127,29,29,.18), var(--shadow);
}}
.ac.hi::before {{
  background:
    radial-gradient(circle at 12% 14%, rgba(251,113,133,.18), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,.05), transparent 28%);
}}
.ac.md {{
  --ac-accent: rgba(251,191,36,.96);
  --ac-soft: rgba(217,119,6,.22);
  background: linear-gradient(135deg, rgba(33,22,8,.92), rgba(16,18,34,.94)) !important;
  border-color: rgba(251,191,36,.24) !important;
  box-shadow: 0 22px 58px rgba(146,64,14,.18), var(--shadow);
}}
.ac.md::before {{
  background:
    radial-gradient(circle at 12% 14%, rgba(251,191,36,.16), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,.05), transparent 28%);
}}
.ac.ok {{
  --ac-accent: rgba(45,212,191,.96);
  --ac-soft: rgba(13,148,136,.18);
  background: linear-gradient(135deg, rgba(7,34,31,.90), rgba(15,18,32,.94)) !important;
  border-color: rgba(45,212,191,.22) !important;
}}
.ac.ok::before {{
  background:
    radial-gradient(circle at 12% 14%, rgba(45,212,191,.16), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,.05), transparent 28%);
}}
.ac.info {{
  --ac-accent: rgba(56,189,248,.96);
  --ac-soft: rgba(14,165,233,.20);
  background: linear-gradient(135deg, rgba(8,22,40,.92), rgba(18,18,42,.94)) !important;
  border-color: rgba(56,189,248,.22) !important;
}}
.ac.info::before {{
  background:
    radial-gradient(circle at 12% 14%, rgba(56,189,248,.16), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,.05), transparent 28%);
}}
.ac-ico-wrap {{
  width: 52px;
  height: 52px;
  min-width: 52px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, color-mix(in srgb, var(--ac-soft) 90%, rgba(255,255,255,.08)), rgba(255,255,255,.03));
  border: 1px solid color-mix(in srgb, var(--ac-accent) 34%, rgba(255,255,255,.08));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 0 24px color-mix(in srgb, var(--ac-accent) 18%, transparent);
  position: relative;
  z-index: 1;
}}
.ac-ico {{
  font-size: 24px;
  line-height: 1;
  color: #FFF7ED;
  filter: drop-shadow(0 0 10px color-mix(in srgb, var(--ac-accent) 52%, transparent));
}}
.ac-main {{
  position: relative;
  z-index: 1;
  flex: 1;
  min-width: 0;
}}
.ac-top {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}}
.ac-eyebrow {{
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: color-mix(in srgb, var(--ac-accent) 70%, #F8FAFC);
}}
.ac-chip {{
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: #F8FAFC;
  background: color-mix(in srgb, var(--ac-soft) 92%, rgba(255,255,255,.06));
  border: 1px solid color-mix(in srgb, var(--ac-accent) 36%, rgba(255,255,255,.08));
  white-space: nowrap;
}}
.ac-ttl {{
  font-weight: 800;
  font-size: 17px;
  letter-spacing: -.2px;
  color: var(--t1);
  margin-bottom: 6px;
}}
.ac-dsc {{
  font-size: 13px;
  color: #B8BDD9;
  line-height: 1.65;
  max-width: 92%;
}}

/* ─── PROGRESS BAR ───────────────────────────────────────────────────────── */
.pb-wrap {{ margin:12px 0; }}
.pb-lbl  {{ display:flex; justify-content:space-between; font-size:12px; color:var(--t2); margin-bottom:5px; }}
.pb-trk  {{ height:9px; background:rgba(255,255,255,.07); border-radius:99px; overflow:hidden; box-shadow:inset 0 1px 8px rgba(0,0,0,.36); }}
.pb-fil  {{ height:100%; border-radius:99px; box-shadow:0 0 18px currentColor; }}

/* ─── STAT ROWS ──────────────────────────────────────────────────────────── */
.sr {{ display:flex; justify-content:space-between; align-items:center; padding:11px 0; border-bottom:1px solid var(--b0); }}
.sr:last-child {{ border:none; }}
.sr-lbl {{ font-size:13px; color:var(--t2); }}
.sr-val {{ font-size:13px; font-weight:600; color:var(--t1); }}
.savings-card {{
  min-height: 172px;
  box-shadow: 0 26px 86px rgba(0,0,0,.50), 0 0 36px rgba(13,148,136,.10);
}}
.savings-rate {{
  font-size:42px;
  font-weight:900;
  letter-spacing:-1.5px;
  line-height:1;
}}

/* ─── CHAT ───────────────────────────────────────────────────────────────── */
.chat-empty {{
  text-align:center; padding:56px 24px; color:var(--t3);
}}
.chat-empty .ce-icon {{ font-size:40px; margin-bottom:14px; }}
.chat-empty .ce-h    {{ font-size:17px; font-weight:700; color:var(--t2); margin-bottom:8px; }}
.chat-empty .ce-sub  {{ font-size:13px; font-weight:300; }}

.msg-u {{ display:flex; justify-content:flex-end; margin:10px 0; }}
.msg-a {{ display:flex; justify-content:flex-start; margin:10px 0; }}
.bub-u {{
  background:linear-gradient(135deg,{VIOLET},{INDIGO});
  color:#fff; border-radius:18px 18px 4px 18px;
  padding:12px 18px; max-width:72%;
  font-size:14px; line-height:1.6;
  box-shadow:0 6px 24px rgba(124,58,237,.35);
}}
.bub-a {{
  background:var(--glass); border:1px solid var(--b0);
  color:var(--t1); border-radius:18px 18px 18px 4px;
  padding:12px 18px; max-width:80%;
  font-size:14px; line-height:1.75;
  backdrop-filter:blur(12px);
  border-left:3px solid {VIOLET2};
}}

/* ─── QA BUTTONS ─────────────────────────────────────────────────────────── */
.stButton > button {{
  background: linear-gradient(135deg,{VIOLET},{INDIGO}) !important;
  color: white !important; border: none !important;
  border-radius: 12px !important;
  font-family: var(--body) !important; font-weight: 600 !important;
  font-size: 13px !important; padding: 9px 18px !important;
  transition: all .2s !important;
  box-shadow: 0 4px 18px rgba(124,58,237,.28) !important;
}}
.stButton > button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 32px rgba(124,58,237,.45) !important;
}}

/* ─── INPUTS ─────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {{
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--b0) !important;
  border-radius: 12px !important;
  color: var(--t1) !important;
  font-family: var(--body) !important;
  font-size: 14px !important;
  padding: 12px 16px !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus {{
  border-color: var(--b1) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,.18) !important;
}}
.stSelectbox > div > div {{
  background: var(--glass) !important; border: 1px solid var(--b0) !important;
  border-radius: 10px !important; color: var(--t1) !important;
}}

/* ─── METRIC WIDGET ──────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {{
  background: var(--glass); border: 1px solid var(--b0);
  border-radius: 16px; padding: 18px 20px;
}}
div[data-testid="stMetricLabel"] > div {{ color:var(--t2) !important; font-size:11px !important; letter-spacing:1px !important; text-transform:uppercase !important; }}
div[data-testid="stMetricValue"]       {{ font-family:var(--body) !important; font-weight:900 !important; color:var(--t1) !important; }}

div[data-testid="stAlert"] {{
  background: linear-gradient(135deg, rgba(14,165,233,.12), rgba(124,58,237,.08)) !important;
  border: 1px solid rgba(14,165,233,.26) !important;
  border-radius: 16px !important;
  box-shadow: 0 16px 48px rgba(0,0,0,.34);
  backdrop-filter: blur(16px);
}}

/* ─── MISC OVERRIDES ─────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {{
  background: var(--glass) !important; border: 1px solid var(--b0) !important;
  border-radius: 14px !important;
}}
hr {{ border-color:var(--b0) !important; opacity:1 !important; }}
.stSpinner > div {{ border-top-color:{VIOLET} !important; }}
header[data-testid="stHeader"] {{ background:transparent !important; }}
.element-container {{ position:relative; z-index:1; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
_PLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#9898C0", size=12),
    margin=dict(l=8, r=8, t=42, b=8),
    legend=dict(bgcolor="rgba(8,8,24,.85)", bordercolor="rgba(255,255,255,.06)",
                borderwidth=1, font=dict(size=12, color="#9898C0")),
    hoverlabel=dict(bgcolor="#0E0E24", bordercolor="rgba(124,58,237,.5)",
                    font=dict(family="Outfit", size=13, color="#F0EFFE")),
    title=dict(font=dict(family="Outfit", size=14, color="#F0EFFE", weight=700),
               x=0, xref="paper", pad=dict(b=12)),
)
_GC = "rgba(255,255,255,0.045)"
_TF = dict(size=11, color="#4A4A70")

def pf(fig: go.Figure, *, xg=True, yg=True) -> go.Figure:
    fig.update_layout(**_PLY)
    fig.update_xaxes(gridcolor=_GC if xg else "rgba(0,0,0,0)",
                     zeroline=False, showline=False, tickfont=_TF)
    fig.update_yaxes(gridcolor=_GC if yg else "rgba(0,0,0,0)",
                     zeroline=False, showline=False, tickfont=_TF)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_df():
    return load_data()

@st.cache_resource(show_spinner=False)
def get_pipe():
    if MODEL_PATH.exists():
        return load_pipeline()
    return train_model(get_df())["pipeline"]

@st.cache_resource(show_spinner=False)
def get_advisor():
    return FinSightAdvisor(finance_context=build_finance_context(get_df()))


def normalize_finance_csv(raw_df: pd.DataFrame):
    # Paste point: add this helper before the SIDEBAR section.
    keywords = {
        "date": ["date", "transaction_date", "time"],
        "expense": ["expense", "amount", "spending", "debit"],
        "income": ["income", "salary", "credit"],
        "category": ["category", "type", "description"],
    }

    work = raw_df.copy()
    work.columns = [str(col).strip() for col in work.columns]
    lookup = {col.lower(): col for col in work.columns}

    def detect_column(target: str):
        for key in keywords[target]:
            if key.lower() in lookup:
                return lookup[key.lower()]
        for normalized, original in lookup.items():
            if any(key.lower() in normalized for key in keywords[target]):
                return original
        return None

    date_col = detect_column("date")
    income_col = detect_column("income")
    expense_col = detect_column("expense") or lookup.get("amount")
    category_col = detect_column("category")

    if date_col is None or expense_col is None:
        return None, "No financial data detected"

    income_values = (
        pd.to_numeric(work[income_col], errors="coerce")
        if income_col is not None
        else pd.Series(np.nan, index=work.index)
    )
    expense_values = pd.to_numeric(work[expense_col], errors="coerce")

    if not (income_values.notna().any() or expense_values.notna().any()):
        return None, "No financial data detected"

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(work[date_col], errors="coerce")
    df["income"] = income_values
    df["expense"] = expense_values
    df["category"] = (
        work[category_col].fillna("Other").astype(str)
        if category_col is not None
        else "Other"
    )

    df["income"] = df["income"].fillna(0)
    df["expense"] = df["expense"].fillna(0)
    df = df.dropna(subset=["date"])

    if df.empty:
        return None, "No financial data detected"

    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["weekday"] = df["date"].dt.day_name()

    return df, None


# ══════════════════════════════════════════════════════════════════════════════
# UI PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════
def page_header(eye: str, h1: str, sub: str = "") -> None:
    st.markdown(f"""
    <div class="ph">
      <div class="ph-eye">{eye}</div>
      <div class="ph-h1">{h1}</div>
      {f'<div class="ph-sub">{sub}</div>' if sub else ""}
    </div>""", unsafe_allow_html=True)

def stitle(t: str) -> None:
    st.markdown(f'<div class="stitle">{t}</div>', unsafe_allow_html=True)

def kpi_card(col, icon, label, value, badge, btype="neu", glow=VIOLET) -> None:
    vcls = "pos" if btype == "up" else ("neg" if btype == "down" else "neu")
    tone = label.lower().replace(" ", "-")
    with col:
        st.markdown(f"""
        <div class="kc glass-card glow-border hover-lift {tone}">
          <div class="kc-glow" style="background:{glow}"></div>
          <span class="kc-icon">{icon}</span>
          <div class="kc-lbl">{label}</div>
          <div class="kc-val metric-value {vcls}">{value}</div>
          <span class="badge {btype}">{badge}</span>
        </div>""", unsafe_allow_html=True)

def pbar(label: str, val_lbl: str, pct: float, color: str = VIOLET) -> None:
    p = min(max(pct * 100, 0), 100)
    st.markdown(f"""
    <div class="pb-wrap">
      <div class="pb-lbl"><span>{label}</span><span>{val_lbl}</span></div>
      <div class="pb-trk"><div class="pb-fil" style="width:{p:.1f}%;background:{color}"></div></div>
    </div>""", unsafe_allow_html=True)

def alert(cls: str, icon: str, title: str, desc: str) -> None:
    label_map = {
        "hi": "High priority",
        "md": "Watchlist",
        "ok": "Healthy",
        "info": "Insight",
    }
    chip = label_map.get(cls, "Alert")
    st.markdown(f"""
    <div class="ac glass-card hover-lift {cls}">
      <div class="ac-bar"></div>
      <div class="ac-ico-wrap"><div class="ac-ico">{icon}</div></div>
      <div class="ac-main">
        <div class="ac-top">
          <div class="ac-eyebrow">Smart signal</div>
          <div class="ac-chip">{chip}</div>
        </div>
        <div class="ac-ttl">{title}</div>
        <div class="ac-dsc">{desc}</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
      <div class="brand">💎 FinSight AI</div>
      <div class="sub">Personal Finance Intelligence</div>
      <div class="live-dot">Live Analytics</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav",
        ["💎  Overview", "📈  Analytics", "🔮  Forecast", "🤖  AI Advisor"],
        label_visibility="collapsed")

    # Paste point: upload block goes below navigation and before Filter Period.
    st.markdown("""
    <style>
      .data-panel-divider {
        height: 1px;
        margin: 22px 4px 18px;
        background: linear-gradient(90deg, transparent, rgba(14,165,233,.55), rgba(167,139,250,.45), transparent);
      }
      .data-panel-heading {
        padding: 0 4px 12px;
      }
      .data-panel-eyebrow {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        color: #4A4A70;
        margin-bottom: 6px;
      }
      .data-panel-title {
        font-size: 16px;
        font-weight: 900;
        color: #F0EFFE;
        line-height: 1.15;
      }
      .upload-card {
        position: relative;
        margin: 0 0 12px;
        padding: 20px 16px 16px;
        border-radius: 20px;
        text-align: center;
        overflow: hidden;
        background:
          linear-gradient(145deg, rgba(14,165,233,.13), rgba(124,58,237,.11)),
          rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 20px 54px rgba(0,0,0,.34), inset 0 1px 0 rgba(255,255,255,.08);
        backdrop-filter: blur(18px) saturate(150%);
        -webkit-backdrop-filter: blur(18px) saturate(150%);
        transition: transform .24s ease, border-color .24s ease, box-shadow .24s ease;
      }
      .upload-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        padding: 1px;
        background: linear-gradient(135deg, rgba(14,165,233,.85), rgba(167,139,250,.72));
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
        opacity: .68;
      }
      .upload-card:hover {
        transform: translateY(-2px);
        border-color: rgba(14,165,233,.34);
        box-shadow: 0 24px 68px rgba(0,0,0,.42), 0 0 30px rgba(14,165,233,.14);
      }
      .upload-icon {
        width: 42px;
        height: 42px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 11px;
        border-radius: 16px;
        color: #E0F2FE;
        background: linear-gradient(135deg, rgba(14,165,233,.24), rgba(167,139,250,.20));
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 0 22px rgba(14,165,233,.14);
        font-size: 21px;
      }
      .upload-title {
        font-size: 15px;
        font-weight: 900;
        color: #F0EFFE;
        margin-bottom: 6px;
      }
      .upload-subtext {
        font-size: 12px;
        line-height: 1.45;
        color: #9898C0;
      }
      .upload-box,
      .upload-success,
      .upload-status {
        border-radius: 16px;
        text-align: center;
        transition: transform .22s ease, border-color .22s ease, background .22s ease, box-shadow .22s ease;
      }
      .upload-success {
        margin: 10px 0 8px;
        padding: 14px 12px;
        color: #D1FAE5;
        background: linear-gradient(135deg, rgba(13,148,136,.16), rgba(14,165,233,.08));
        border: 1px solid rgba(45,212,191,.26);
        box-shadow: 0 16px 38px rgba(0,0,0,.24), 0 0 20px rgba(45,212,191,.08);
      }
      .upload-success-title {
        font-size: 13px;
        font-weight: 850;
        margin-bottom: 5px;
      }
      .upload-success-meta {
        font-size: 11px;
        color: #99F6E4;
        opacity: .9;
      }
      .upload-status {
        margin: 10px 0 8px;
        padding: 11px 12px;
        color: #C7D2FE;
        background: rgba(255,255,255,.052);
        border: 1px solid rgba(255,255,255,.10);
        font-size: 12px;
        font-weight: 650;
      }
      section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        padding: 4px 0 2px;
      }
      section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label {
        display: none;
      }
      section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
        position: relative;
        min-height: 118px;
        padding: 0;
        border-radius: 18px;
        overflow: hidden;
        background:
          linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.025)),
          rgba(5,8,22,.54);
        border: 1px dashed rgba(14,165,233,.42);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.07);
      }
      section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] * {
        visibility: hidden;
      }
      section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"]::before {
        content: "☁";
        visibility: visible;
        position: absolute;
        top: 24px;
        left: 50%;
        transform: translateX(-50%);
        width: 32px;
        height: 32px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #E0F2FE;
        background: rgba(14,165,233,.14);
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 0 18px rgba(14,165,233,.12);
        font-size: 17px;
      }
      section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"]::after {
        content: "Drop your CSV file here or click to browse";
        visibility: visible;
        position: absolute;
        left: 18px;
        right: 18px;
        top: 66px;
        color: #DDE7FF;
        font-size: 12px;
        font-weight: 750;
        line-height: 1.35;
        text-align: center;
      }
      section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"]:hover {
        transform: translateY(-1px);
        border-color: rgba(167,139,250,.68);
        background: linear-gradient(145deg, rgba(14,165,233,.08), rgba(167,139,250,.07));
        box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 0 24px rgba(14,165,233,.11);
      }
      .upload-title {
        font-size: 0;
      }
      .upload-title::before {
        content: "📂\\A Upload Financial Data";
        white-space: pre;
        display: block;
        font-size: 15px;
        line-height: 1.45;
      }
      .upload-subtext {
        font-size: 0;
      }
      .upload-subtext::before {
        content: "Import your transactions and unlock insights";
        font-size: 12px;
      }
    </style>
    <div class="data-panel-divider"></div>
    <div class="data-panel-heading">
      <div class="data-panel-eyebrow">Control Center</div>
      <div class="data-panel-title">💼 Data Panel</div>
    </div>
    <div class="upload-card">
      <div class="upload-title">📂 Upload Your Financial Data</div>
      <div class="upload-subtext">Drag & drop or browse your CSV file</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="upload-status">Upload your financial data from the sidebar to get started</div>',
            unsafe_allow_html=True,
        )

    df_full = get_df()
    if uploaded_file is not None:
        try:
            uploaded_raw = pd.read_csv(uploaded_file)
            uploaded_df, upload_error = normalize_finance_csv(uploaded_raw)
            if upload_error:
                st.error(upload_error)
            else:
                df_full = uploaded_df
                st.markdown(
                    f"""
                    <div class="upload-success">
                      <div class="upload-success-title">✅ Data successfully loaded</div>
                      <div class="upload-success-meta">{len(df_full):,} rows · {uploaded_file.name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            st.error("No financial data detected")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#4A4A70;padding:0 4px 4px">Filter Period</p>', unsafe_allow_html=True)
    dmin, dmax = df_full["date"].min().date(), df_full["date"].max().date()
    sd = st.date_input("From", dmin, dmin, dmax, label_visibility="collapsed")
    ed = st.date_input("To",   dmax, dmin, dmax, label_visibility="collapsed")

    df = df_full[(df_full["date"].dt.date >= sd) & (df_full["date"].dt.date <= ed)].copy()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:11px;color:#4A4A70;padding:0 4px">{len(df):,} records · {sd:%d %b %Y} → {ed:%d %b %Y}</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 ─ OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "💎  Overview":
    page_header("FINANCIAL OVERVIEW", "Your Wealth Command Center",
                "Live snapshot of income, expenses, savings health, and smart alerts.")

    kpis    = compute_kpis(df)
    monthly = get_monthly_summary(df)
    cat     = get_category_summary(df)
    alerts  = detect_overspending(df)
    adv_alerts = advanced_alerts(df)
    insights=generate_insights(df)
    opp     = savings_opportunity_report(df)

    # ── 4 KPI Cards ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    sr, ns = kpis["savings_rate"], kpis["net_savings"]
    kpi_card(c1, "💰", "Total Income",   fmt_inr(kpis["total_income"],  compact=True),
             "This Period", "neu", TEAL)
    kpi_card(c2, "📤", "Total Expenses", fmt_inr(kpis["total_expense"], compact=True),
             f"Avg {fmt_inr(kpis['avg_daily_expense'])}/day", "down", ROSE)
    kpi_card(c3, "📈", "Net Savings",    fmt_inr(ns, compact=True),
             "Surplus" if ns >= 0 else "Deficit", "up" if ns >= 0 else "down", SKY)
    kpi_card(c4, "🎯", "Savings Rate",   fmt_pct(sr),
             f"Target ≥ {fmt_pct(HEALTHY_SAVINGS_RATE)}",
             "up" if sr >= HEALTHY_SAVINGS_RATE else "down", VIOLET)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cash Flow Trend + Donut Pie ─────────────────────────────────────────
    ca, cb = st.columns([3, 2], gap="large")

    with ca:
        stitle("📊 Monthly Cash Flow")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month_year"], y=monthly["total_income"],
            name="Income", mode="lines",
            line=dict(color=SKY, width=2.5),
            fill="tozeroy", fillcolor="rgba(14,165,233,.08)",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["month_year"], y=monthly["total_expense"],
            name="Expenses", mode="lines+markers",
            line=dict(color=ROSE, width=2, dash="dot"),
            marker=dict(size=5, color=ROSE),
        ))
        fig.add_trace(go.Bar(
            x=monthly["month_year"], y=monthly["net_savings"],
            name="Net Savings", yaxis="y2", opacity=.65,
            marker_color=[TEAL if v >= 0 else ROSE for v in monthly["net_savings"]],
        ))
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        tickfont=dict(size=10, color="#4A4A70")),
            hovermode="x unified", xaxis=dict(showgrid=False),
        )
        st.plotly_chart(pf(fig), use_container_width=True)

    with cb:
        stitle("🍩 Spending Distribution")
        fig2 = go.Figure(go.Pie(
            labels=cat["category"], values=cat["total"], hole=.6,
            marker=dict(colors=PAL, line=dict(color="rgba(2,3,10,.92)", width=2)),
            textinfo="label+percent", textfont=dict(size=12, color="#D7D9F5"),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f} · %{percent}<extra></extra>",
            pull=[.035 if i == 0 else 0 for i in range(len(cat))],
            sort=False,
        ))
        fig2.update_traces(
            opacity=.94,
            hoverlabel=dict(bgcolor="#071226", bordercolor="rgba(56,189,248,.28)")
        )
        fig2.update_layout(showlegend=True, legend=dict(
            orientation="h", yanchor="bottom", y=-.18, xanchor="center", x=.5,
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=11, color="#AEB4D8")
        ), annotations=[dict(
            text=f"<b>{fmt_inr(cat['total'].sum(), compact=True)}</b><br><span style='font-size:11px'>TOTAL</span>",
            x=.5, y=.5, showarrow=False, align="center",
            font=dict(family="Outfit", color="#F0EFFE", size=18),
        )])
        st.plotly_chart(pf(fig2, xg=False, yg=False), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Smart Alerts + Savings Health ──────────────────────────────────────
    cc, cd = st.columns(2, gap="large")

    with cc:
        stitle("⚡ Smart Alerts")
        if not alerts:
            alert("ok", "OK", "All clear - spending within limits",
                  "No category exceeds 20% of your total expenses.")
        for a in alerts:
            cls  = "hi" if a["severity"] == "high" else "md"
            icon = "!" if a["severity"] == "high" else "~"
            alert(cls, icon,
                  f"{a['category']} - {a['share_pct']}% of spend",
                  f"Total: {fmt_inr(a['total'])} | Avg {fmt_inr(a['avg_daily'])}/day | {a['severity'].upper()}")
        for a in adv_alerts:
            alert("md", "!", "Momentum alert", a)
    st.markdown("<br>", unsafe_allow_html=True)
    insights = generate_insights(df)
    stitle("🧠 Smart Insights")

    if not insights:
        st.info("No major issues detected. You're doing good 👍")
    else:
        for i in insights:
            alert("info", "i", "Smart insight", i)

    with cd:
        stitle("💚 Savings Health Score")
        tc = TEAL if opp["on_track"] else ROSE
        st.markdown(f"""
        <div class="gp glass-card glow-border hover-lift savings-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
            <div>
              <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--t3)">Savings Rate</div>
              <div class="metric-value savings-rate" style="color:{tc};">{fmt_pct(opp['actual_rate'])}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:10px;color:var(--t3)">Target</div>
              <div style="font-size:26px;font-weight:700;color:var(--t3)">{fmt_pct(opp['target_rate'])}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        pbar("Progress toward target", fmt_pct(opp["actual_rate"]),
             opp["actual_rate"] / max(opp["target_rate"], 0.001), tc)
        pos_m = opp["total_months"] - opp["negative_months"]
        st.markdown(f"""
        <div style="margin-top:14px">
          <div class="sr"><span class="sr-lbl">🟢 Positive months</span><span class="sr-val" style="color:{TEAL}">{pos_m}/{opp['total_months']}</span></div>
          <div class="sr"><span class="sr-lbl">🔴 Deficit months</span><span class="sr-val" style="color:{ROSE}">{opp['negative_months']}/{opp['total_months']}</span></div>
          <div class="sr"><span class="sr-lbl">🏆 Best month</span><span class="sr-val">{opp['best_month']}</span></div>
          <div class="sr"><span class="sr-lbl">📉 Worst month</span><span class="sr-val">{opp['worst_month']}</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Daily Expense Sparkline ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    stitle("📅 Daily Expense Rhythm")
    dfs = df.sort_values("date").copy()
    dfs["roll7"] = dfs["expense"].rolling(7).mean()
    figsp = go.Figure()
    figsp.add_trace(go.Scatter(
        x=dfs["date"], y=dfs["expense"], mode="lines", name="Daily",
        line=dict(color=VIOLET, width=1.3),
        fill="tozeroy", fillcolor="rgba(124,58,237,.07)",
        hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>",
    ))
    figsp.add_trace(go.Scatter(
        x=dfs["date"], y=dfs["roll7"], mode="lines", name="7-day avg",
        line=dict(color=AMBER, width=2, dash="dash"),
        hovertemplate="%{x|%d %b}<br>Avg ₹%{y:,.0f}<extra></extra>",
    ))
    figsp.update_layout(hovermode="x unified")
    st.plotly_chart(pf(figsp), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 ─ ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Analytics":
    page_header("SPENDING INTELLIGENCE", "Deep-Dive Analytics",
                "Granular category breakdown, time patterns, and income composition.")

    cat     = get_category_summary(df)
    src     = get_income_source_summary(df)
    monthly = get_monthly_summary(df)

    # ── Category Leaderboard ───────────────────────────────────────────────
    stitle("📊 Category Leaderboard")
    fig_lb = go.Figure()
    for i, row in cat.iterrows():
        fig_lb.add_trace(go.Bar(
            x=[row["total"]], y=[row["category"]], orientation="h",
            marker=dict(color=PAL[i % len(PAL)], opacity=.85, line=dict(width=0)),
            text=f"  ₹{row['total']:,.0f}  ({row['share_pct']}%)",
            textposition="outside", textfont=dict(size=12, color="#9898C0"),
            showlegend=False,
            hovertemplate=f"<b>{row['category']}</b><br>₹{row['total']:,.0f}<br>{row['share_pct']}%<extra></extra>",
        ))
    fig_lb.update_layout(barmode="overlay", height=440, showlegend=False)
    fig_lb.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(pf(fig_lb, yg=False), use_container_width=True)

    # ── Heatmap + Income Sources ───────────────────────────────────────────
    ch, ci = st.columns([3, 2], gap="large")
    with ch:
        stitle("🗓 Category × Month Heatmap")
        pivot = df.pivot_table(values="expense", index="category",
                               columns="month_name", aggfunc="sum", fill_value=0)
        mo = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot.reindex(columns=[m for m in mo if m in pivot.columns])
        fig_hm = px.imshow(pivot,
            color_continuous_scale=[[0,"#0A0A20"],[.35,"#3B0764"],[.7,VIOLET],[1,ROSE]],
            aspect="auto")
        fig_hm.update_traces(
            text=[[f"₹{v:,.0f}" for v in r] for r in pivot.values],
            texttemplate="%{text}", textfont=dict(size=10, color="rgba(255,255,255,.65)"),
        )
        fig_hm.update_coloraxes(showscale=False)
        st.plotly_chart(pf(fig_hm, xg=False, yg=False), use_container_width=True)

    with ci:
        stitle("💼 Income Composition")
        if not src.empty:
            fig_src = go.Figure(go.Bar(
                x=src["income"], y=src["income_source"], orientation="h",
                marker=dict(color=[SKY, TEAL, AMBER, VIOLET][:len(src)],
                            opacity=.85, line=dict(width=0)),
                text=[fmt_inr(v, compact=True) for v in src["income"]],
                textposition="outside", textfont=dict(size=12, color="#9898C0"),
                hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
            ))
            fig_src.update_layout(showlegend=False)
            fig_src.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(pf(fig_src, yg=False), use_container_width=True)
        else:
            st.info("No income data for this period.")

    # ── Waterfall ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    stitle("🌊 Monthly Net Savings Waterfall")

    fig_wf = go.Figure(go.Waterfall(
        x=monthly["month_year"],
        y=monthly["net_savings"],
        measure=["relative"] * len(monthly),
        connector=dict(
            line=dict(color="rgba(255,255,255,.08)", width=1)
        ),
        increasing=dict(marker=dict(color=TEAL)),
        decreasing=dict(marker=dict(color=ROSE)),
        totals=dict(marker=dict(color=VIOLET)),
        text=[fmt_inr(v, compact=True) for v in monthly["net_savings"]],
        textposition="outside",
        textfont=dict(size=11, color="#9898C0"),
        hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
    ))
    st.plotly_chart(pf(fig_wf), use_container_width=True)

    # ── Weekday Heatmap ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    stitle("📆 Weekday Spending Pattern")
    wd_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wday = (df.groupby("weekday")["expense"].mean().reset_index()
              .assign(weekday=lambda d: pd.Categorical(d["weekday"], wd_order, True))
              .sort_values("weekday"))
    wday["short"] = wday["weekday"].str[:3]
    fig_wd = go.Figure(go.Bar(
        x=wday["short"], y=wday["expense"],
        marker=dict(color=wday["expense"],
                    colorscale=[[0,"#160830"],[.5,VIOLET],[1,ROSE]], line=dict(width=0)),
        text=[f"₹{v:,.0f}" for v in wday["expense"]],
        textposition="outside", textfont=dict(size=11, color="#9898C0"),
        hovertemplate="<b>%{x}</b><br>Avg ₹%{y:,.0f}<extra></extra>",
        showlegend=False,
    ))
    fig_wd.update_layout(yaxis_title="Average Daily Expense (₹)")
    st.plotly_chart(pf(fig_wd), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 ─ FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Forecast":
    page_header("ML PREDICTION ENGINE", "Expense Forecasting",
                "Gradient Boosting model predicts your daily expenses using temporal patterns.")

    with st.spinner("Loading prediction model …"):
        pipe = get_pipe()

    # ── Controls ───────────────────────────────────────────────────────────
    st.markdown('<div class="gp">', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([1, 1, 1], gap="medium")
    with fc1:
        pred_year  = st.selectbox("📅 Year",  [2026, 2027, 2028])
    with fc2:
        pred_month = st.selectbox("🗓 Month", range(1, 13),
                                  format_func=lambda m: date(2000, m, 1).strftime("%B"))
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Run Forecast", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        with st.spinner("Computing forecast …"):
            pred_df = predict_month_expenses(pred_year, pred_month, pipe)

        tp   = pred_df["predicted_expense"].sum()
        ap   = pred_df["predicted_expense"].mean()
        peak = pred_df.loc[pred_df["predicted_expense"].idxmax()]

        st.markdown("<br>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3, gap="medium")
        kpi_card(pc1, "📦", "Predicted Monthly Total", fmt_inr(tp, compact=True), "Full Month", "neu", VIOLET)
        kpi_card(pc2, "📊", "Daily Average",            fmt_inr(ap),              "Per Day",    "neu", SKY)
        kpi_card(pc3, "🔺", "Peak Spend Day",
                 peak["date"].strftime("%d %b"),
                 fmt_inr(peak["predicted_expense"]), "down", ROSE)

        st.markdown("<br>", unsafe_allow_html=True)
        stitle("📈 Day-by-Day Forecast")

        noise = pred_df["predicted_expense"].std() * 0.38
        pred_df["hi"] = pred_df["predicted_expense"] + noise
        pred_df["lo"] = np.maximum(0, pred_df["predicted_expense"] - noise)
        xl = pred_df["date"].dt.strftime("%d %b")

        figf = go.Figure()
        figf.add_trace(go.Scatter(
            x=list(xl) + list(xl[::-1]),
            y=list(pred_df["hi"]) + list(pred_df["lo"][::-1]),
            fill="toself", fillcolor="rgba(124,58,237,.08)",
            line=dict(width=0), name="Confidence Band",
            hoverinfo="skip", showlegend=True,
        ))
        figf.add_trace(go.Scatter(
            x=xl, y=pred_df["predicted_expense"],
            mode="lines+markers", name="Predicted",
            line=dict(color=VIOLET, width=2.5),
            marker=dict(size=6, color=VIOLET, line=dict(color="#03030A", width=2)),
            hovertemplate="%{x}<br><b>₹%{y:,.0f}</b><extra></extra>",
        ))
        figf.add_hline(y=ap, line=dict(color=AMBER, width=1.5, dash="dot"),
                       annotation_text=f"Avg {fmt_inr(ap)}",
                       annotation_font=dict(color=AMBER, size=11))
        figf.update_layout(hovermode="x unified", yaxis_title="Expense (₹)")
        st.plotly_chart(pf(figf), use_container_width=True)

        with st.expander("📋 Full Day-wise Table"):
            disp = pred_df[["date","predicted_expense"]].copy()
            disp["date"] = disp["date"].dt.strftime("%A, %d %b %Y")
            disp["predicted_expense"] = disp["predicted_expense"].map(lambda x: f"₹{x:,.0f}")
            st.dataframe(disp.rename(columns={"date":"Date","predicted_expense":"Predicted Expense"}),
                         use_container_width=True, hide_index=True)

    # ── Retrain + Eval ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    stitle("⚙️ Model Evaluation")
    if st.button("🔄 Re-train & Evaluate"):
        with st.spinner("Training on full dataset …"):
            res = train_model(get_df())
        m = res["metrics"]
        ec1, ec2, ec3, ec4 = st.columns(4, gap="medium")
        for col, lbl, val, tip in [
            (ec1, "MAE",    f"₹{m['mae']:,.0f}",   "Mean Absolute Error"),
            (ec2, "RMSE",   f"₹{m['rmse']:,.0f}",  "Root Mean Squared Error"),
            (ec3, "R²",     f"{m['r2']:.3f}",       "Coefficient of Determination"),
            (ec4, "CV-MAE", f"₹{m['cv_mae']:,.0f}", "5-fold Cross-validated MAE"),
        ]:
            with col:
                st.metric(lbl, val, help=tip)

        st.markdown("<br>", unsafe_allow_html=True)
        ev1, ev2 = st.columns(2, gap="large")
        with ev1:
            stitle("🎛 Feature Importance")
            fi = (pd.DataFrame(list(m["feature_importance"].items()),
                               columns=["Feature","Importance"])
                  .sort_values("Importance", ascending=True))
            figfi = go.Figure(go.Bar(
                x=fi["Importance"], y=fi["Feature"], orientation="h",
                marker=dict(color=fi["Importance"],
                            colorscale=[[0,"#160830"],[.5,VIOLET],[1,SKY]], line=dict(width=0)),
                text=[f"{v:.3f}" for v in fi["Importance"]],
                textposition="outside", textfont=dict(size=11, color="#9898C0"),
                showlegend=False,
            ))
            figfi.update_layout(showlegend=False)
            st.plotly_chart(pf(figfi, yg=False), use_container_width=True)

        with ev2:
            stitle("🎯 Actual vs Predicted")
            yt, yp = res["y_test"].values, res["y_pred"]
            mn, mx = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            figap = go.Figure()
            figap.add_trace(go.Scatter(
                x=yt, y=yp, mode="markers",
                marker=dict(color=VIOLET, opacity=.5, size=6,
                            line=dict(color="#03030A", width=1)),
                name="Predictions",
                hovertemplate="Actual ₹%{x:,.0f}<br>Pred ₹%{y:,.0f}<extra></extra>",
            ))
            figap.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(color=TEAL, dash="dash", width=1.5), name="Perfect Fit",
            ))
            figap.update_layout(xaxis_title="Actual (₹)", yaxis_title="Predicted (₹)")
            st.plotly_chart(pf(figap), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 ─ AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  AI Advisor":
    page_header("POWERED BY CLAUDE", "Your Personal Finance AI",
                "Context-aware advice grounded in your real transaction data — not generic tips.")

    # Session state
    if "chat" not in st.session_state:
        st.session_state.chat = []

    if "adv" not in st.session_state:
        with st.spinner("Initialising AI advisor with your financial profile …"):
            st.session_state.adv = FinSightAdvisor(
                build_finance_context(df), df
            )

    adv: FinSightAdvisor = st.session_state.adv

    # ── Quick Action Chips ─────────────────────────────────────────────────
    stitle("⚡ Instant Insights")
    qa_map = {
        "📋 Health Check":   "Give me a concise financial health summary with exact numbers and a rating out of 10.",
        "✂️ Cut Spending":   "Which 3 expense categories should I reduce first? Show the ₹ impact of a 25% cut each.",
        "💰 Savings Plan":   "Build a realistic 90-day plan to push my savings rate to 20%. Be specific.",
        "📅 Worst Month":    "Analyse my worst spending month — what drove it and how can I avoid repeating it?",
        "🧾 Category Deep":  "Do a deep-dive on my top expense category: trends, anomalies, and reduction tactics.",
        "📆 Next Month Tip": "Based on my patterns, what is the single most impactful thing I can do next month?",
    }
    rows = [list(qa_map.items())[:3], list(qa_map.items())[3:]]
    for row_items in rows:
        cols = st.columns(3, gap="small")
        for col, (lbl, prompt) in zip(cols, row_items):
            if col.button(lbl, use_container_width=True, key=f"qa_{lbl}"):
                st.session_state.chat.append(("user", prompt))
                with st.spinner("FinSight AI is thinking …"):
                    st.session_state.chat.append(("assistant", adv.chat(prompt)))
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    stitle("💬 Conversation")

    # ── Chat Render ────────────────────────────────────────────────────────
    if not st.session_state.chat:
        st.markdown("""
        <div class="chat-empty">
          <div class="ce-icon">✦</div>
          <div class="ce-h">Ready to analyse your finances</div>
          <div class="ce-sub">Tap a Quick Action above, or type your question below.<br>
          I have full context of your income, expenses, and saving patterns.</div>
        </div>""", unsafe_allow_html=True)
    else:
        for role, content in st.session_state.chat:
            if role == "user":
                st.markdown(f'<div class="msg-u"><div class="bub-u">{content}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="msg-a"><div class="bub-a">', unsafe_allow_html=True)
                st.markdown(content)
                st.markdown('</div></div>', unsafe_allow_html=True)

    # ── Input Row ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("cf", clear_on_submit=True):
        i1, i2 = st.columns([6, 1], gap="small")
        q = i1.text_input("q", placeholder="Ask anything … e.g. How much did I overspend on food this year?",
                          label_visibility="collapsed")
        send = i2.form_submit_button("Send →", use_container_width=True)

    if send and q.strip():
        st.session_state.chat.append(("user", q.strip()))
        with st.spinner("FinSight is analysing …"):
            st.session_state.chat.append(("assistant", adv.chat(q.strip())))
        st.rerun()

    # ── Bottom Row ─────────────────────────────────────────────────────────
    if st.session_state.chat:
        _, clr = st.columns([5, 1])
        if clr.button("🗑 Clear", use_container_width=True):
            st.session_state.chat = []
            adv.reset()
            st.rerun()

    # ── Financial context preview ──────────────────────────────────────────
    with st.expander("🔍 View Financial Context injected into AI"):
        st.code(build_finance_context(df), language="text")

    st.markdown("""
    <div style="text-align:center;margin-top:40px;font-size:10px;color:#2A2A45;letter-spacing:2px">
      FINSIGHT AI  ·  CLAUDE-POWERED  ·  FOR INFORMATIONAL PURPOSES ONLY
    </div>""", unsafe_allow_html=True)



    
