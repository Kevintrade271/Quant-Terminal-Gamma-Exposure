# ============================================================
# SPY – Gamma / Delta / Vanna + Absolute Gamma + Combo Strikes
# Plotly (dark), intradía con toggles y FULLSCREEN HTML.py
# ============================================================
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# --------------------------
# Parámetros de usuario
# --------------------------
N_EXPIRIES        = 4        # 0DTE + próximas (si hay)
MULTIPLIER        = 100      # contrato US
DEFAULT_SWEEP     = 0.01     # rango inicial ±1%
WALL_QUANTILE     = 0.90
EXPORT_HTML_FS    = True
HTML_PATH         = "spy_options_dashboard_fullscreen.html"

# --------------------------
# Black–Scholes (vector)
# --------------------------
def d1_d2(S, K, T, r, q, sigma):
    S = np.asarray(S, float); K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), 5e-4)
    sigma = np.maximum(np.asarray(sigma, float), 1e-6)
    vol_sqrt = sigma * np.sqrt(T)
    mu = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
    d1 = mu / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2

def delta_bs(S, K, T, r, q, sigma, cp):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * np.asarray(T, float))
    base = disc_q * norm.cdf(d1)
    cp = np.asarray(cp)
    return np.where(cp == 1, base, disc_q * (norm.cdf(d1) - 1.0))

def gamma_bs(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    denom = (np.asarray(S, float) * np.asarray(sigma, float) * np.sqrt(np.asarray(T, float))) + 1e-12
    return np.exp(-q * np.asarray(T, float)) * norm.pdf(d1) / denom

def vanna_dDelta_dSigma(S, K, T, r, q, sigma, cp, eps=1e-4):
    sigma = np.asarray(sigma, float)
    up = delta_bs(S, K, T, r, q, sigma + eps, cp)
    dn = delta_bs(S, K, T, r, q, sigma - eps, cp)
    return (up - dn) / (2.0 * eps)

def yearfrac(expiry_ts, now_utc=None):
    if now_utc is None:
        now_utc = pd.Timestamp(datetime.now(timezone.utc))
    expiry_ts = pd.Timestamp(expiry_ts).tz_localize(None)
    now_utc   = pd.Timestamp(now_utc).tz_localize(None)
    dt_days = (expiry_ts - now_utc).total_seconds()/(24*3600)
    return max(dt_days/365.0, 5e-4)

# --------------------------
# Descargar SPY y cadenas
# --------------------------
tk = yf.Ticker("SPY")
S0 = float(tk.fast_info["last_price"])

# r (13w T-Bill ^IRX) y q (dividend yield aprox)
try:
    r_y = yf.Ticker("^IRX").fast_info.get("last_price", None)
    r = float(r_y)/100.0 if r_y is not None else 0.05
except Exception:
    r = 0.05
try:
    q_guess = tk.fast_info.get("dividend_yield", None)
    q = float(q_guess) if q_guess is not None else 0.015
except Exception:
    q = 0.015

exps = tk.options
if not exps: raise SystemExit("No hay expiraciones disponibles.")
exps = exps[:N_EXPIRIES]

frames = []
for exp in exps:
    ch = tk.option_chain(exp)
    for side, cp in [("calls", +1), ("puts", -1)]:
        df = getattr(ch, side).copy()
        if df.empty: continue
        df["cp"] = cp
        df["expiry"] = pd.to_datetime(exp)
        frames.append(df)

opt = pd.concat(frames, ignore_index=True)
opt = opt.rename(columns={"impliedVolatility":"iv", "openInterest":"open_interest"})
opt = opt.dropna(subset=["iv","open_interest"])
opt["T"] = opt["expiry"].apply(yearfrac)
opt = opt[(opt["open_interest"]>0) & (opt["T"]>0)]

# --------------------------
# Gamma por strike (niveles estilo SpotGamma)
# --------------------------
opt["gamma"]    = gamma_bs(S=S0, K=opt["strike"], T=opt["T"], r=r, q=q, sigma=opt["iv"])
opt["GEX_line"] = opt["gamma"] * (S0**2) * opt["open_interest"] * MULTIPLIER
gex_calls = opt[opt["cp"]==+1].groupby("strike")["GEX_line"].sum()
gex_puts  = opt[opt["cp"]==-1].groupby("strike")["GEX_line"].sum()
gex_net   = gex_calls.add(gex_puts, fill_value=0).sort_index()

cum = (gex_net - gex_net.mean()).cumsum()
gamma_flip = float(cum.sub(0).abs().idxmin())
thr = gex_net.quantile(WALL_QUANTILE)
gamma_walls = gex_net[gex_net >= thr].sort_values(ascending=False)

call_wall = float(gex_calls.idxmax()) if not gex_calls.empty else None
put_wall  = float(gex_puts.abs().idxmax()) if not gex_puts.empty else None
key_gamma = float(gex_net.idxmax()) if not gex_net.empty else None

# AbsGamma y Combo (Total / Next)
abs_gamma_by_strike = opt.groupby("strike")["GEX_line"].apply(lambda s: np.abs(s).sum()).sort_index()
abs_gamma_by_exp = opt.groupby("expiry")["GEX_line"].apply(lambda s: np.abs(s).sum()).sort_values(ascending=False)
next_expiry = abs_gamma_by_exp.index[0]
abs_gamma_next = (opt[opt["expiry"]==next_expiry]
                  .groupby("strike")["GEX_line"].apply(lambda s: np.abs(s).sum()).sort_index())
combo_total = gex_calls.add(-np.abs(gex_puts), fill_value=0).sort_index()
combo_next = (opt[opt["expiry"]==next_expiry]
              .assign(sign=lambda d: np.where(d["cp"]==1, 1, -1))
              .groupby("strike")
              .apply(lambda d: (d["GEX_line"]*d["sign"]).sum())
              .sort_index())

# --------------------------
# Δ y Vanna vs spot – precomputar ±1%, ±2%, ±5%
# --------------------------
def book_delta_at(S):
    d = delta_bs(S=S, K=opt["strike"], T=opt["T"], r=r, q=q, sigma=opt["iv"], cp=opt["cp"])
    return np.sum(d * opt["open_interest"].values * MULTIPLIER)

def book_vanna_at(S):
    v = vanna_dDelta_dSigma(S=S, K=opt["strike"], T=opt["T"], r=r, q=q, sigma=opt["iv"], cp=opt["cp"])
    return np.sum(v * opt["open_interest"].values * MULTIPLIER)

sweeps = [0.01, 0.02, 0.05]  # botones
grids = {}
delta_curves = {}
vanna_curves = {}
for p in sweeps:
    grid = np.linspace(S0*(1-p), S0*(1+p), 121)
    grids[p] = grid
    delta_curves[p] = np.array([book_delta_at(S) for S in grid])
    vanna_curves[p] = np.array([book_vanna_at(S) for S in grid])

# --------------------------
# FIGURA (2 filas x 3 columnas) + botones
# --------------------------
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        "SPY — Gamma Model (GEX por strike)",
        "SPY — Delta Model (Δ agregado vs Spot)",
        "SPY — Vanna Model (∂Δ/∂σ agregado vs Spot)",
        "SPY — Absolute Gamma (Total / Next)",
        f"SPY — Combo Strikes (Total / Next) | Next: {next_expiry.date()}",
        ""
    )
)

# --- Fila 1: Gamma ---
fig.add_trace(
    go.Scatter(x=gex_net.index, y=gex_net.values, mode="lines",
               name="GEX neto", line=dict(width=2, color="#2dd4bf"),
               hovertemplate="Strike=%{x}<br>GEX=%{y:,.0f}<extra></extra>"),
    row=1, col=1
)

def add_vline_gamma(x, name, color="#94a3b8", dash="dash"):
    if x is None: return
    fig.add_vline(x=x, line_width=1, line_dash=dash, line_color=color, row=1, col=1)
    fig.add_annotation(x=x, y=max(gex_net.values), text=name, showarrow=False,
                       yshift=14, font=dict(color="#cbd5e1", size=10), row=1, col=1)

add_vline_gamma(call_wall, "Call Wall")
add_vline_gamma(put_wall,  "Put Wall")
add_vline_gamma(key_gamma, "Key Gamma")
add_vline_gamma(gamma_flip,"VT/Flip", color="#ffffff", dash="dot")

# --- Fila 1: Delta (3 trazas, visible sólo la de ±1%)
for i,p in enumerate(sweeps):
    fig.add_trace(
        go.Scatter(x=grids[p], y=delta_curves[p], mode="lines",
                   name=f"Δ ±{int(p*100)}%", line=dict(width=2, color="#f59e0b"),
                   hovertemplate="Spot=%{x:.2f}<br>Δ=%{y:,.0f}<extra></extra>",
                   visible=(p==DEFAULT_SWEEP)),
        row=1, col=2
    )
# Líneas guía (Spot/Flip) para Delta
for x in [S0, gamma_flip]:
    fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="#ffffff", row=1, col=2)

# --- Fila 1: Vanna (3 trazas, visible sólo ±1%)
for i,p in enumerate(sweeps):
    fig.add_trace(
        go.Scatter(x=grids[p], y=vanna_curves[p], mode="lines",
                   name=f"Vanna ±{int(p*100)}%", line=dict(width=2, color="#a78bfa"),
                   hovertemplate="Spot=%{x:.2f}<br>Vanna=%{y:,.0f}<extra></extra>",
                   visible=(p==DEFAULT_SWEEP)),
        row=1, col=3
    )
# Líneas guía (Spot/Flip) para Vanna
for x in [S0, gamma_flip]:
    fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="#ffffff", row=1, col=3)

# --- Fila 2: Absolute Gamma y Combo (Total/Next – ambas ocultables)
abs_total_trace = go.Bar(x=abs_gamma_by_strike.index, y=abs_gamma_by_strike.values,
                         name="AbsGamma Total", marker_color="#22d3ee",
                         hovertemplate="Strike=%{x}<br>AbsGamma=%{y:,.0f}<extra></extra>",
                         visible=True)
abs_next_trace  = go.Bar(x=abs_gamma_next.index, y=abs_gamma_next.values,
                         name="AbsGamma Next", marker_color="#fb7185", opacity=0.55,
                         hovertemplate="Strike=%{x}<br>AbsGamma Next=%{y:,.0f}<extra></extra>",
                         visible=False)
combo_total_trace= go.Bar(x=combo_total.index, y=combo_total.values,
                          name="Combo Total (Calls+ / Puts-)", marker_color="#60a5fa",
                          hovertemplate="Strike=%{x}<br>Combo=%{y:,.0f}<extra></extra>",
                          visible=True)
combo_next_trace = go.Bar(x=combo_next.index, y=combo_next.values,
                          name="Combo Next", marker_color="#f97316", opacity=0.55,
                          hovertemplate="Strike=%{x}<br>Combo Next=%{y:,.0f}<extra></extra>",
                          visible=False)

fig.add_trace(abs_total_trace, row=2, col=1)
fig.add_trace(abs_next_trace,  row=2, col=1)
fig.add_trace(combo_total_trace,row=2, col=2)
fig.add_trace(combo_next_trace, row=2, col=2)

# --------------------------
# Layout y ejes
# --------------------------
fig.update_layout(
    template="plotly_dark",
    title=f"SPY Options Dashboard — Spot {S0:.2f} | r={100*r:.2f}% | q={100*q:.2f}%",
    autosize=True, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=0.02),
    margin=dict(l=20, r=20, t=60, b=20)
)

# Ejes fila 1
fig.update_xaxes(title_text="Strike", row=1, col=1)
fig.update_yaxes(title_text="GEX (≈ γ·S²·OI·mult)", row=1, col=1)
# Por defecto, usar rango de ±1% (se actualizará con botones)
fig.update_xaxes(title_text="Spot", row=1, col=2, range=[S0*(1-DEFAULT_SWEEP*1.05), S0*(1+DEFAULT_SWEEP*1.05)])
fig.update_yaxes(title_text="Delta nocional (sum OI)", row=1, col=2)
fig.update_xaxes(title_text="Spot", row=1, col=3, range=[S0*(1-DEFAULT_SWEEP*1.05), S0*(1+DEFAULT_SWEEP*1.05)])
fig.update_yaxes(title_text="Vanna nocional (sum OI)", row=1, col=3)

# Ejes fila 2
fig.update_xaxes(title_text="Strike", row=2, col=1)
fig.update_yaxes(title_text="|Gamma| nocional", row=2, col=1)
fig.update_xaxes(title_text="Strike", row=2, col=2)
fig.update_yaxes(title_text="Gamma neta (calls+ / puts-)", row=2, col=2)

# --------------------------
# Botones (updatemenus)
# --------------------------
# Mapeo de índices de trazas:
# 0 = GEX neto
# 1..3 = Delta ±1/±2/±5
# 4..6 = Vanna ±1/±2/±5
# 7,8 = AbsGamma Total / Next
# 9,10 = Combo Total / Next
n_traces = 11

def vis_all_false():
    return [False]*n_traces

def set_visibility(delta_pick, vanna_pick, show_total, show_next):
    """
    delta_pick, vanna_pick ∈ {0,1,2} -> (±1%, ±2%, ±5%)
    show_total/show_next -> bool
    """
    vis = [True] + [False]*(n_traces-1)  # siempre mostrar GEX (traza 0)
    # Delta
    vis[1 + delta_pick] = True
    # Vanna
    vis[4 + vanna_pick] = True
    # Abs/Combo
    vis[7] = show_total
    vis[8] = show_next
    vis[9] = show_total
    vis[10]= show_next
    return vis

# Botones de Total/Next/Both para Abs/Combo
buttons_abs_combo = [
    dict(label="Total", method="update",
         args=[{"visible": set_visibility(0,0,True,False)},
               {}]),
    dict(label="Next", method="update",
         args=[{"visible": set_visibility(0,0,False,True)},
               {}]),
    dict(label="Ambos", method="update",
         args=[{"visible": set_visibility(0,0,True,True)},
               {}]),
]

# Botones de rango para Delta/Vanna
def btn_range(pct, delta_idx, vanna_idx):
    xr = [S0*(1-pct*1.05), S0*(1+pct*1.05)]
    return dict(
        label=f"±{int(pct*100)}%",
        method="update",
        args=[{"visible": set_visibility(delta_idx, vanna_idx,
                                        fig.data[7].visible, fig.data[8].visible)},
              {"xaxis2.range": xr, "xaxis3.range": xr}]
    )

buttons_range = [
    btn_range(0.01, 0, 0),
    btn_range(0.02, 1, 1),
    btn_range(0.05, 2, 2),
]

# Botón de Zoom Gamma (±3% o full)
gamma_full_range = [min(gex_net.index), max(gex_net.index)]
gamma_zoom_range = [S0*(1-0.03), S0*(1+0.03)]
buttons_gamma = [
    dict(label="Gamma Zoom (±3%)", method="relayout",
         args=[{"xaxis.range": gamma_zoom_range}]),
    dict(label="Gamma Full", method="relayout",
         args=[{"xaxis.range": gamma_full_range}]),
]

fig.update_layout(
    updatemenus=[
        dict(type="buttons", direction="left", x=0.01, y=-0.10,
             buttons=buttons_abs_combo, bgcolor="#1f2937", bordercolor="#374151"),
        dict(type="buttons", direction="left", x=0.44, y=-0.10,
             buttons=buttons_range, bgcolor="#1f2937", bordercolor="#374151"),
        dict(type="buttons", direction="left", x=0.82, y=-0.10,
             buttons=buttons_gamma, bgcolor="#1f2937", bordercolor="#374151"),
    ]
)

# Mostrar
fig.show(config={"responsive": True, "displaylogo": False})

# ---- HTML Fullscreen 100% viewport
if EXPORT_HTML_FS:
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>SPY Options Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  html,body,#app {{ height:100%; width:100%; margin:0; background:#111; }}
  #chart {{ height:100vh; width:100vw; }}
</style>
</head>
<body>
<div id="app"><div id="chart"></div></div>
<script>
  var fig = {fig.to_json()};
  Plotly.newPlot('chart', fig.data, fig.layout, {{responsive:true, displaylogo:false}});
  window.addEventListener('resize', ()=>Plotly.Plots.resize(document.getElementById('chart')));
</script>
</body>
</html>
"""
    Path(HTML_PATH).write_text(html, encoding="utf-8")
    print(f"Dashboard FULLSCREEN guardado en: {HTML_PATH}")

# Resumen útil en consola
print(f"\nSpot: {S0:.2f}")
print(f"Call Wall: {call_wall} | Put Wall: {put_wall} | Key Gamma: {key_gamma} | VT/Flip ~ {gamma_flip:.0f}")
