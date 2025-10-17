#!/usr/bin/env python3
# trace_style_plotly_final.py
# ------------------------------------------------------------
# Perfil TRACE-like en Plotly. Zoom ±2%, colores fijos (Puts=Rojo, Calls=Morado),
# y escala robusta del eje X para una visualización profesional.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timezone
import plotly.graph_objects as go
from pathlib import Path

# --- Estilo visual profesional ---
# No es necesario si solo usamos Plotly, pero lo mantenemos por si se reutiliza
import matplotlib.pyplot as plt
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.facecolor"] = "#0f1115"
plt.rcParams["figure.facecolor"] = "#0f1115"
plt.rcParams["savefig.facecolor"] = "#0f1115"
plt.rcParams["text.color"] = "#e5e7eb"
plt.rcParams["axes.labelcolor"] = "#e5e7eb"
plt.rcParams["xtick.color"] = "#d1d5db"
plt.rcParams["ytick.color"] = "#d1d5db"

# --- Black–Scholes y Utilidades ---
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

def build_greeks_df(ticker="SPY", max_exp=6, r=0.045, q=0.012, min_oi=200, max_bas=np.inf):
    tk = yf.Ticker(ticker)
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        hist = tk.history(period="1d")
        if hist is None or hist.empty: raise RuntimeError("No se pudo obtener el spot.")
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
            df = df[["strike", "impliedVolatility", "openInterest", "bid", "ask"]].dropna()
            if min_oi > 0: df = df[df.openInterest >= min_oi]
            if np.isfinite(max_bas):
                bas = (df["ask"] - df["bid"]).abs()
                df = df[(bas > 0) & (bas <= max_bas)]
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

def export_fullscreen_html(fig, filename="dashboard.html"):
    title = fig.layout.title.text
    html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"/><title>{title}</title><script src="https://cdn.plot.ly/plotly-latest.min.js"></script><style>html,body,#chart{{height:100%;width:100%;margin:0;padding:0;overflow:hidden;}}body{{background-color:#111827;}}</style></head><body><div id="chart"></div><script>var fig={fig.to_json()};Plotly.newPlot('chart',fig.data,fig.layout,{{responsive:true,displaylogo:false}});window.addEventListener('resize',()=>Plotly.Plots.resize(document.getElementById('chart')));</script></body></html>"""
    Path(filename).write_text(html, encoding="utf-8")
    print(f"Dashboard interactivo guardado en: {filename}")

# --- Función de Gráfico Principal ---
def plot_trace_style_plotly(df, spot, ticker="SPY", value="GEX", exp="nearest", zoom_pct=0.02):
    if df.empty:
        raise SystemExit("Sin datos para graficar.")

    exps_sorted = sorted(df["exp"].unique(), key=lambda d: pd.to_datetime(d))
    chosen = exps_sorted[0] if exp == "nearest" else pd.to_datetime(exp).date()
    if chosen not in df["exp"].unique():
        raise SystemExit(f"Exp {exp} no disponible. Opciones: {exps_sorted}")

    sub = df[df["exp"] == chosen].copy()
    
    k_min, k_max = spot * (1 - zoom_pct), spot * (1 + zoom_pct)
    sub = sub[(sub['K'] >= k_min) & (sub['K'] <= k_max)]
    if sub.empty:
        print(f"Advertencia: No hay datos de opciones en el rango de ±{zoom_pct*100:.0f}% para la exp {chosen}.")
        return None

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

    fig.add_hline(y=spot, line_width=1.5, line_dash="dash", line_color="#facc15", annotation_text="ATM", annotation_position="bottom right")

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
        height=800,
        margin=dict(l=70, r=20, t=60, b=40)
    )
    return fig

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    TICKER, ZOOM_PERCENT = "SPY", 0.02
    
    try:
        spot, df = build_greeks_df(TICKER, max_exp=8, r=0.05, q=0.015, min_oi=100)
        
        print("Generando perfil GEX...")
        fig_gex = plot_trace_style_plotly(df, spot, ticker=TICKER, value="GEX", exp="nearest", zoom_pct=ZOOM_PERCENT)
        if fig_gex:
            fig_gex.show(config={'displaylogo': False})
            export_fullscreen_html(fig_gex, filename=f"{TICKER}_gex_profile.html")
        
        print("\nGenerando perfil CHARM...")
        fig_charm = plot_trace_style_plotly(df, spot, ticker=TICKER, value="CHARM", exp="nearest", zoom_pct=ZOOM_PERCENT)
        if fig_charm:
            fig_charm.show(config={'displaylogo': False})
            export_fullscreen_html(fig_charm, filename=f"{TICKER}_charm_profile.html")
            
    except Exception as e:
        print(f"Error al ejecutar el script: {e}")