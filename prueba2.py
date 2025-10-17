#!/usr/bin/env python3
# full_dashboard_live.py
# ------------------------------------------------------------
# Dashboard completo de GEX, Delta, Vanna, etc., en Plotly/Dash con auto-actualización.
# ------------------------------------------------------------
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Parámetros de Usuario ---
TICKER = "SPY"
N_EXPIRIES = 4
MULTIPLIER = 100
DEFAULT_SWEEP = 0.01
WALL_QUANTILE = 0.90
UPDATE_INTERVAL_MIN = 5

# --- Lógica de Black–Scholes y Datos (reutilizada) ---
def d1_d2(S, K, T, r, q, sigma):
    S, K = np.asarray(S, float), np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), 5e-4)
    sigma = np.maximum(np.asarray(sigma, float), 1e-6)
    vol_sqrt = sigma * np.sqrt(T)
    mu = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
    d1 = mu / vol_sqrt
    return d1, d1 - vol_sqrt

def delta_bs(S, K, T, r, q, sigma, cp):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * np.asarray(T, float))
    base = disc_q * norm.cdf(d1)
    return np.where(np.asarray(cp) == 1, base, disc_q * (norm.cdf(d1) - 1.0))

def gamma_bs(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    denom = (np.asarray(S, float) * sigma * np.sqrt(T)) + 1e-12
    return np.exp(-q * T) * norm.pdf(d1) / denom

def vanna_dDelta_dSigma(S, K, T, r, q, sigma, cp, eps=1e-4):
    up = delta_bs(S, K, T, r, q, sigma + eps, cp)
    dn = delta_bs(S, K, T, r, q, sigma - eps, cp)
    return (up - dn) / (2.0 * eps)

def yearfrac(expiry_ts, now_utc=None):
    now_utc = now_utc or pd.Timestamp(datetime.now(timezone.utc))
    dt_days = (pd.Timestamp(expiry_ts).tz_localize(None) - now_utc.tz_localize(None)).total_seconds() / (24*3600)
    return max(dt_days / 365.0, 5e-4)

def fetch_and_process_data():
    tk = yf.Ticker(TICKER)
    spot = float(tk.fast_info["last_price"])
    try:
        r = float(yf.Ticker("^IRX").fast_info.get("last_price", 5.0)) / 100.0
    except:
        r = 0.05
    try:
        q = float(tk.fast_info.get("dividend_yield", 0.015))
    except:
        q = 0.015

    exps = tk.options[:N_EXPIRIES]
    frames = []
    for exp in exps:
        ch = tk.option_chain(exp)
        for side, cp in [("calls", 1), ("puts", -1)]:
            df = getattr(ch, side).copy()
            if df.empty: continue
            df["cp"] = cp
            df["expiry"] = pd.to_datetime(exp)
            frames.append(df)
    
    opt = pd.concat(frames, ignore_index=True).rename(columns={"impliedVolatility":"iv", "openInterest":"open_interest"})
    opt = opt.dropna(subset=["iv", "open_interest"])
    opt["T"] = opt["expiry"].apply(yearfrac)
    opt = opt[(opt["open_interest"] > 0) & (opt["T"] > 0)].copy()

    # Calcular Griegas y Exposiciones
    opt["gamma"] = gamma_bs(spot, opt["strike"], opt["T"], r, q, opt["iv"])
    opt["GEX_line"] = opt["gamma"] * (spot**2) * opt["open_interest"] * MULTIPLIER
    
    gex_calls = opt[opt["cp"] == 1].groupby("strike")["GEX_line"].sum()
    gex_puts = opt[opt["cp"] == -1].groupby("strike")["GEX_line"].sum()
    gex_net = gex_calls.add(gex_puts, fill_value=0).sort_index()
    
    # Calcular Niveles Clave
    cum = (gex_net - gex_net.mean()).cumsum()
    gamma_flip = float(cum.sub(0).abs().idxmin())
    call_wall = float(gex_calls.idxmax()) if not gex_calls.empty else None
    put_wall = float(gex_puts.idxmax()) if not gex_puts.empty else None # Puts have negative GEX, so max is least negative/most positive dealer exposure
    
    abs_gamma_by_strike = opt.groupby("strike")["GEX_line"].apply(lambda s: np.abs(s).sum()).sort_index()
    combo_total = gex_calls.add(-np.abs(gex_puts), fill_value=0).sort_index()
    
    # Pre-calcular curvas de Delta y Vanna
    sweeps = [0.01, 0.02, 0.05]
    grids, delta_curves, vanna_curves = {}, {}, {}
    for p in sweeps:
        grid = np.linspace(spot * (1-p), spot * (1+p), 51)
        grids[p] = grid
        delta_curves[p] = np.array([np.sum(delta_bs(S, opt["strike"], opt["T"], r, q, opt["iv"], opt["cp"]) * opt["open_interest"] * MULTIPLIER) for S in grid])
        vanna_curves[p] = np.array([np.sum(vanna_dDelta_dSigma(S, opt["strike"], opt["T"], r, q, opt["iv"], opt["cp"]) * opt["open_interest"] * MULTIPLIER) for S in grid])

    return {
        "spot": spot, "r": r, "q": q, "gex_net": gex_net, "gamma_flip": gamma_flip,
        "call_wall": call_wall, "put_wall": put_wall, "abs_gamma": abs_gamma_by_strike,
        "combo": combo_total, "sweeps": sweeps, "grids": grids,
        "delta_curves": delta_curves, "vanna_curves": vanna_curves
    }

# --- Función para Crear la Figura de Plotly ---
def create_dashboard_figure(data):
    fig = make_subplots(rows=2, cols=3, subplot_titles=("Gamma Model", "Delta Model", "Vanna Model", "Absolute Gamma", "Combo Strikes", ""))
    
    # Fila 1: GEX, Delta, Vanna
    fig.add_trace(go.Scatter(x=data['gex_net'].index, y=data['gex_net'].values, name="GEX Net", line=dict(color="#2dd4bf")), row=1, col=1)
    
    # Add Vlines
    for k, v in {"Call Wall": data['call_wall'], "Put Wall": data['put_wall'], "Flip": data['gamma_flip']}.items():
        if v:
            fig.add_vline(x=v, line_width=1, line_dash="dash", line_color="grey", annotation_text=k, row=1, col=1)
            fig.add_vline(x=v, line_width=1, line_dash="dash", line_color="grey", row=1, col=2)
            fig.add_vline(x=v, line_width=1, line_dash="dash", line_color="grey", row=1, col=3)

    for p in data['sweeps']:
        vis = (p == DEFAULT_SWEEP)
        fig.add_trace(go.Scatter(x=data['grids'][p], y=data['delta_curves'][p], name=f"Delta ±{int(p*100)}%", visible=vis, line=dict(color="#f59e0b")), row=1, col=2)
        fig.add_trace(go.Scatter(x=data['grids'][p], y=data['vanna_curves'][p], name=f"Vanna ±{int(p*100)}%", visible=vis, line=dict(color="#a78bfa")), row=1, col=3)
        
    # Fila 2: Absolute Gamma, Combo
    fig.add_trace(go.Bar(x=data['abs_gamma'].index, y=data['abs_gamma'].values, name="Absolute Gamma", marker_color="#22d3ee"), row=2, col=1)
    fig.add_trace(go.Bar(x=data['combo'].index, y=data['combo'].values, name="Combo Strikes", marker_color="#60a5fa"), row=2, col=2)

    # Layout y Botones
    fig.update_layout(
        template="plotly_dark",
        title_text=f"{TICKER} Options Dashboard | Spot: {data['spot']:.2f}",
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
        height=800
    )
    # Ejes
    fig.update_xaxes(title_text="Strike", row=1, col=1)
    fig.update_yaxes(title_text="GEX", row=1, col=1)
    fig.update_xaxes(title_text="Spot Price", row=1, col=2)
    fig.update_yaxes(title_text="Total Delta", row=1, col=2)
    fig.update_xaxes(title_text="Spot Price", row=1, col=3)
    fig.update_yaxes(title_text="Total Vanna", row=1, col=3)
    fig.update_xaxes(title_text="Strike", row=2, col=1)
    fig.update_yaxes(title_text="|Gamma|", row=2, col=1)
    fig.update_xaxes(title_text="Strike", row=2, col=2)
    fig.update_yaxes(title_text="Gamma (C+ / P-)", row=2, col=2)

    # Botones
    buttons = []
    for i, p in enumerate(data['sweeps']):
        visibility = [True] # GEX trace
        for j in range(len(data['sweeps'])):
            visibility.append(i == j) # Delta traces
        for j in range(len(data['sweeps'])):
            visibility.append(i == j) # Vanna traces
        visibility.extend([True, True]) # AbsGamma and Combo traces
        
        buttons.append(dict(
            label=f"±{int(p*100)}%",
            method="update",
            args=[{"visible": visibility}]
        ))
        
    fig.update_layout(
        updatemenus=[
            dict(type="buttons", direction="right", x=0.5, y=1.1, xanchor="center", yanchor="top",
                 buttons=buttons)
        ]
    )
    return fig

# --- Construcción de la Aplicación Dash ---
app = dash.Dash(__name__)
app.title = f"{TICKER} Options Dashboard"

app.layout = html.Div(style={'backgroundColor': '#111827'}, children=[
    html.Div(id='live-update-text', style={'color': '#9ca3af', 'textAlign': 'center', 'padding': '10px'}),
    dcc.Graph(id='options-dashboard-graph', style={'height': '95vh'}),
    dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL_MIN * 60 * 1000, n_intervals=0)
])

@app.callback(
    [Output('options-dashboard-graph', 'figure'),
     Output('live-update-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Actualizando datos...")
        data = fetch_and_process_data()
        fig = create_dashboard_figure(data)
        update_time = f"Última Actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print("Actualización completada.")
        return fig, update_time
    except Exception as e:
        print(f"Error durante la actualización: {e}")
        error_fig = go.Figure().update_layout(
            template="plotly_dark", 
            title_text=f"Error al obtener datos: {e}. Reintentando...",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return error_fig, f"Error: {e}"

# --- Ejecutar el Servidor ---
if __name__ == '__main__':
    app.run(debug=True, port=8080)