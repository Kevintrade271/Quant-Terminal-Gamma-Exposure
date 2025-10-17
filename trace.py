#!/usr/bin/env python3
# trace_dashboard_live_port_8090.py
# ------------------------------------------------------------
# Dashboard TRACE-like en Plotly/Dash con actualización automática en el puerto 8090.
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

# --- Lógica de Cálculo de Griegas y Datos ---
def bs_d1(S, K, T, r, q, iv):
    return (np.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * np.sqrt(T) + 1e-12)

def bs_gamma(S, K, T, r, q, iv):
    d1 = bs_d1(S, K, T, r, q, iv)
    return np.exp(-q * T) * norm.pdf(d1) / (S * iv * np.sqrt(T) + 1e-12)

def bs_charm(S, K, T, r, q, iv, cp):
    d1 = bs_d1(S, K, T, r, q, iv)
    d2 = d1 - iv * np.sqrt(T)
    s = 1 if cp == "C" else -1
    t1 = -q * np.exp(-q * T) * norm.cdf(s * d1) * s
    t2 = np.exp(-q * T) * norm.pdf(d1) * (2 * (r - q) * T - d2 * iv * np.sqrt(T)) / (2 * T * iv * np.sqrt(T) + 1e-12)
    return t1 + t2

def _load_chain(ticker, exp):
    tk = yf.Ticker(ticker)
    oc = tk.option_chain(exp)
    for df in (oc.calls, oc.puts):
        for c in ["impliedVolatility", "strike", "openInterest", "bid", "ask"]:
            if c not in df.columns: df[c] = np.nan
    return oc

def build_greeks_df(ticker="SPY", max_exp=6, r=0.045, q=0.012, min_oi=200):
    tk = yf.Ticker(ticker)
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        hist = tk.history(period="1d")
        if hist.empty: raise RuntimeError("No se pudo obtener el spot.")
        spot = float(hist["Close"].iloc[-1])
    exps = tk.options[:max_exp] if tk.options else []
    if not exps: raise RuntimeError("Sin expiraciones disponibles.")
    now = pd.Timestamp.utcnow().tz_localize(None)
    rows = []
    for e in exps:
        exp_ts = pd.to_datetime(e)
        T = max((exp_ts - now).total_seconds(), 0) / (365.0 * 24 * 3600.0)
        if T <= 0: continue
        try:
            oc = _load_chain(ticker, e)
        except Exception:
            continue
        for side, df in (("C", oc.calls), ("P", oc.puts)):
            if df is None or df.empty: continue
            df = df[["strike", "impliedVolatility", "openInterest"]].dropna()
            if min_oi > 0: df = df[df.openInterest >= min_oi]
            df = df[(df.impliedVolatility > 0) & (df.strike > 0)]
            if df.empty: continue
            for _, r0 in df.iterrows():
                K=float(r0.strike); iv=float(r0.impliedVolatility); oi=float(r0.openInterest)
                gamma = bs_gamma(spot, K, T, r, q, iv)
                charm = bs_charm(spot, K, T, r, q, iv, side)
                mult = 100.0
                gex = -oi * gamma * (spot * mult)
                chm = -oi * charm * mult
                rows.append({"exp": pd.to_datetime(e).date(), "K": K, "side": side, "GEX": gex, "CHARM": chm})
    return spot, pd.DataFrame(rows)

# --- Función de Gráfico Principal ---
def plot_trace_style_plotly(df, spot, ticker="SPY", value="GEX", exp="nearest", zoom_pct=0.02):
    if df.empty: return go.Figure().update_layout(template="plotly_dark", title_text=f"Sin datos para {value}")
    exps_sorted = sorted(df["exp"].unique(), key=lambda d: pd.to_datetime(d))
    chosen = exps_sorted[0] if exp == "nearest" else pd.to_datetime(exp).date()
    if chosen not in df["exp"].unique(): return go.Figure().update_layout(template="plotly_dark", title_text=f"Expiración no encontrada para {value}")

    sub = df[df["exp"] == chosen].copy()
    k_min, k_max = spot * (1 - zoom_pct), spot * (1 + zoom_pct)
    sub = sub[(sub['K'] >= k_min) & (sub['K'] <= k_max)]
    if sub.empty: return go.Figure().update_layout(template="plotly_dark", title_text=f"Sin datos en el rango de zoom para {value}")

    all_strikes_in_range = np.sort(sub['K'].unique())
    g_c = sub.groupby("K")[value].sum().where(sub.groupby("K")['side'].first() == 'C', 0).reindex(all_strikes_in_range, fill_value=0)
    g_p = sub.groupby("K")[value].sum().where(sub.groupby("K")['side'].first() == 'P', 0).reindex(all_strikes_in_range, fill_value=0)
    
    left_vals, right_vals = -g_p, g_c
    PUT_COLOR, CALL_COLOR = "#ef4444", "#8b5cf6"
    
    topP_idx = np.argsort(np.abs(left_vals.values))[-3:]
    topC_idx = np.argsort(np.abs(right_vals.values))[-3:]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=all_strikes_in_range, x=left_vals, orientation='h', name='Puts', marker_color=PUT_COLOR, hovertemplate='<b>%{y} P</b><br>'+f'{value}: '+'%{x:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(y=all_strikes_in_range, x=right_vals, orientation='h', name='Calls', marker_color=CALL_COLOR, hovertemplate='<b>%{y} C</b><br>'+f'{value}: '+'%{x:,.0f}<extra></extra>'))
    
    for side, top_indices, vals, x_align in [("P", topP_idx, left_vals, -1), ("C", topC_idx, right_vals, 1)]:
        for rank, idx in enumerate(np.flip(top_indices), start=1):
            if idx >= len(all_strikes_in_range): continue
            k, v = all_strikes_in_range[idx], vals.iloc[idx]
            bar_color = PUT_COLOR if side == 'P' else CALL_COLOR
            fig.add_trace(go.Bar(y=[k], x=[v], orientation='h', showlegend=False, marker=dict(color=bar_color, line=dict(color='white', width=1.5)), hovertemplate=f'<b>{k} {side} (Top {rank})</b><br>{value}: {v:,.0f}<extra></extra>'))
            fig.add_annotation(y=k, x=v, text=f"{side}{rank}", showarrow=False, xshift=12 * x_align, font=dict(color="white", size=10, family="Arial Black"))

    fig.add_hline(y=spot, line_width=1.5, line_dash="dash", line_color="#facc15", annotation_text=f"ATM {spot:.2f}", annotation_position="bottom right")

    top_p_vals = left_vals.iloc[topP_idx].abs()
    top_c_vals = right_vals.iloc[topC_idx].abs()
    max_p = top_p_vals.max() if not top_p_vals.empty and not top_p_vals.isnull().all() else 0
    max_c = top_c_vals.max() if not top_c_vals.empty and not top_c_vals.isnull().all() else 0
    xmax_val = max(max_p, max_c)
    xmax = xmax_val * 1.25 if xmax_val > 0 else 1

    fig.update_layout(
        template="plotly_dark",
        title=f"{value} Profile (Split) — {ticker} @ {spot:.2f} | Exp: {chosen}",
        barmode='relative',
        yaxis_title='Strike',
        xaxis_title=f"{value} (Dealer Exposure)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=[-xmax, xmax]),
        yaxis=dict(range=[k_max, k_min], autorange=False, type='linear'),
        margin=dict(l=70, r=20, t=60, b=40)
    )
    return fig

# --- Construcción de la Aplicación Dash ---
app = dash.Dash(__name__)
app.title = "SPY Options Profile"

app.layout = html.Div(style={'backgroundColor': '#111827', 'padding': '10px'}, children=[
    html.H1("SPY GEX & CHARM Dashboard", style={'color': '#e5e7eb', 'textAlign': 'center'}),
    html.Div(id='live-update-text', style={'color': '#9ca3af', 'textAlign': 'center', 'marginBottom': '20px'}),
    dcc.Graph(id='gex-profile-graph', style={'height': '85vh'}),
    dcc.Graph(id='charm-profile-graph', style={'height': '85vh', 'marginTop': '20px'}),
    dcc.Interval(
        id='interval-component',
        interval=5 * 60 * 1000,
        n_intervals=0
    )
])

@app.callback(
    [Output('gex-profile-graph', 'figure'),
     Output('charm-profile-graph', 'figure'),
     Output('live-update-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Actualizando datos de SPY...")
        spot, df = build_greeks_df(ticker="SPY", max_exp=8, r=0.05, q=0.015, min_oi=100)
        
        fig_gex = plot_trace_style_plotly(df, spot, ticker="SPY", value="GEX", exp="nearest", zoom_pct=0.02)
        fig_charm = plot_trace_style_plotly(df, spot, ticker="SPY", value="CHARM", exp="nearest", zoom_pct=0.02)
        
        update_time = f"Última Actualización: {datetime.now().strftime('%H:%M:%S')}"
        print("Actualización completada.")
        return fig_gex, fig_charm, update_time
    except Exception as e:
        print(f"Error durante la actualización: {e}")
        error_fig = go.Figure().update_layout(
            template="plotly_dark", 
            title_text=f"Error al obtener datos. Reintentando...",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return error_fig, error_fig, f"Error: {e}"

# --- Ejecutar el Servidor ---
if __name__ == '__main__':
    ### CAMBIO QTP: Especificar el puerto 8090 ###
    app.run(debug=True, port=8090)