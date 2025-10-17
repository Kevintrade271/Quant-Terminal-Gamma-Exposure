#!/usr/bin/env python3
# vol_dashboard_live_v2.py
# ------------------------------------------------------------
# Dashboard interactivo con límite de strikes para legibilidad.
# ------------------------------------------------------------
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime

# --- Lógica de Cálculo de Griegas y Datos ---
def z_last(hist_vals, now):
    x = np.array(list(hist_vals) + [now], dtype=float)
    if x.std() < 1e-9: return 0.0
    return float((x - x.mean())[-1] / x.std())

def get_vix_data():
    vix_tk = yf.Ticker("^VIX")
    vix_hist = vix_tk.history(period="6mo")
    if vix_hist.empty: raise RuntimeError("No se pudo obtener el histórico del VIX.")
    current_vix = float(vix_hist["Close"].iloc[-1])
    return vix_hist["Close"], current_vix

def safe_mean(a, b):
    vals = [x for x in (a, b) if (pd.notna(x) and x > 0)]
    return float(np.mean(vals)) if vals else np.nan

def nearest(arr, value):
    a = np.asarray(arr, dtype=float)
    return a[(np.abs(a - value)).argmin()]

def load_and_build_matrix(ticker="SPY", max_exp=15, strike_span=0.10, max_cols=40, min_oi=100):
    tk = yf.Ticker(ticker)
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        hist = tk.history(period="1d")
        if hist.empty: raise RuntimeError("No se pudo obtener el spot.")
        spot = float(hist["Close"].iloc[-1])

    exps = tk.options[:max_exp] if tk.options else []
    if not exps: raise RuntimeError("Sin expiraciones disponibles.")
    
    rows = []
    lo_k, hi_k = spot * (1 - strike_span), spot * (1 + strike_span)
    for e in exps:
        try:
            oc = tk.option_chain(e)
            calls_df, puts_df = oc.calls.copy(), oc.puts.copy()
        except Exception:
            continue
        for side_name, df in (("calls", calls_df), ("puts", puts_df)):
            for c in ["impliedVolatility", "strike", "openInterest"]:
                if c not in df.columns: df[c] = np.nan
            df = df.dropna(subset=["impliedVolatility", "strike", "openInterest"])
            if min_oi > 0: df = df[df["openInterest"] >= min_oi]
            df = df[(df["strike"] >= lo_k) & (df["strike"] <= hi_k)]
            if side_name == "calls": calls = df
            else: puts = df
        if calls.empty and puts.empty: continue
        c = calls[["strike", "impliedVolatility"]].rename(columns={"impliedVolatility": "iv_c"})
        p = puts[["strike", "impliedVolatility"]].rename(columns={"impliedVolatility": "iv_p"})
        m = pd.merge(c, p, on="strike", how="outer")
        m["iv"] = m.apply(lambda r: safe_mean(r["iv_c"], r["iv_p"]), axis=1)
        m = m.dropna(subset=["iv"])
        if m.empty: continue
        m["exp"] = pd.to_datetime(e).date()
        rows.append(m[["exp", "strike", "iv"]])
    
    if not rows: return pd.DataFrame(), spot
    
    big = pd.concat(rows, ignore_index=True)
    
    ### CAMBIO QTP: Limitar el número de strikes (columnas) para legibilidad ###
    strikes = np.sort(big["strike"].unique())
    if len(strikes) > max_cols:
        atm_k = nearest(strikes, spot)
        # Seleccionar los 'max_cols' strikes más cercanos al precio spot
        order = np.argsort(np.abs(strikes - atm_k))[:max_cols]
        keep_strikes = np.sort(strikes[order])
        big = big[big["strike"].isin(keep_strikes)]
        
    pivot = big.pivot_table(index="exp", columns="strike", values="iv", aggfunc="mean").sort_index().sort_index(axis=1)
    return pivot, spot

# --- Función para Crear la Figura de Plotly ---
def create_heatmap_figure(pivot_df, spot, vix_zscore, ticker):
    if pivot_df.empty:
        return go.Figure().update_layout(template="plotly_dark", title_text="Esperando datos...")

    data = pivot_df.values * 100.0
    
    if vix_zscore > 0.5:
        colorscale = "Greens"
        color_theme = "Miedo/Oportunidad"
    elif vix_zscore < -0.5:
        colorscale = "Reds"
        color_theme = "Complacencia/Peligro"
    else:
        colorscale = "Greys"
        color_theme = "Neutral"

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=colorscale,
        colorbar={"title": "IV (%)"}
    ))

    annotations = []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if np.isnan(val): continue
            
            ### CAMBIO QTP: Lógica de contraste de texto mejorada ###
            # Usamos un umbral fijo para decidir el color del texto. Esto es más estable.
            text_color = "#111827" if val > 30 else "#f9fafb" # Ejemplo: IV > 30% -> fondo oscuro -> texto claro
            
            annotations.append(
                dict(
                    x=pivot_df.columns[j],
                    y=pivot_df.index[i],
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color=text_color, size=9, family="Arial") # Tamaño de fuente ligeramente mayor
                )
            )
            
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title_text = f"{ticker} Volatility Dashboard | Spot: {spot:.2f} | VIX Z-Score: {vix_zscore:+.2f} ({color_theme}) | Última Actualización: {now}"

    fig.update_layout(
        template="plotly_dark",
        title=title_text,
        xaxis_title="Strike",
        yaxis_title="Expiración",
        yaxis=dict(autorange="reversed", type='category'), # 'category' para asegurar espaciado uniforme
        annotations=annotations,
        height=800
    )
    return fig

# --- Construcción de la Aplicación Dash ---
app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#111827'}, children=[
    dcc.Graph(id='volatility-heatmap', style={'height': '98vh'}),
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,
        n_intervals=0
    )
])

@app.callback(
    Output('volatility-heatmap', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_heatmap(n):
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Actualizando datos...")
        vix_hist, current_vix = get_vix_data()
        vix_zscore = z_last(vix_hist.iloc[:-1].values, current_vix)
        
        pivot_data, spot_price = load_and_build_matrix(ticker="SPY")
        
        fig = create_heatmap_figure(pivot_data, spot_price, vix_zscore, "SPY")
        print("Actualización completada.")
        return fig
    except Exception as e:
        print(f"Error durante la actualización: {e}")
        return go.Figure().update_layout(
            template="plotly_dark", 
            title_text=f"Error al actualizar datos: {e}. Reintentando en 5 minutos...",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

# --- Ejecutar el Servidor ---
if __name__ == '__main__':
    app.run(debug=True, port=9999)