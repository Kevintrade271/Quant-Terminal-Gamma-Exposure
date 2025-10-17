#!/usr/bin/env python3

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

TICKER = "SPY"
REFRESH_INTERVAL = 5 * 60 * 1000

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

def delta_bs(S, K, T, r, q, sigma, cp):
    d1 = bs_d1(S, K, T, r, q, sigma)
    disc_q = np.exp(-q * np.asarray(T, float))
    base = disc_q * norm.cdf(d1)
    cp = np.asarray(cp)
    return np.where(cp == 1, base, disc_q * (norm.cdf(d1) - 1.0))

def vanna_dDelta_dSigma(S, K, T, r, q, sigma, cp, eps=1e-4):
    sigma = np.asarray(sigma, float)
    up = delta_bs(S, K, T, r, q, sigma + eps, cp)
    dn = delta_bs(S, K, T, r, q, sigma - eps, cp)
    return (up - dn) / (2.0 * eps)

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
        margin=dict(l=70, r=20, t=60, b=40),
        height=700
    )
    return fig

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
    strikes = np.sort(big["strike"].unique())
    if len(strikes) > max_cols:
        atm_k = nearest(strikes, spot)
        order = np.argsort(np.abs(strikes - atm_k))[:max_cols]
        keep_strikes = np.sort(strikes[order])
        big = big[big["strike"].isin(keep_strikes)]

    pivot = big.pivot_table(index="exp", columns="strike", values="iv", aggfunc="mean").sort_index().sort_index(axis=1)
    return pivot, spot

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
            text_color = "#111827" if val > 30 else "#f9fafb"
            annotations.append(
                dict(
                    x=pivot_df.columns[j],
                    y=pivot_df.index[i],
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color=text_color, size=9, family="Arial")
                )
            )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title_text = f"{ticker} Volatility Dashboard | Spot: {spot:.2f} | VIX Z-Score: {vix_zscore:+.2f} ({color_theme}) | Última Actualización: {now}"

    fig.update_layout(
        template="plotly_dark",
        title=title_text,
        xaxis_title="Strike",
        yaxis_title="Expiración",
        yaxis=dict(autorange="reversed", type='category'),
        annotations=annotations,
        height=700
    )
    return fig

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Quant Terminal - Unified Dashboard"

app.layout = html.Div(style={'backgroundColor': '#0a0e17', 'minHeight': '100vh'}, children=[
    html.H1("🎯 Quant Terminal - Dashboard Unificado",
            style={'color': '#e5e7eb', 'textAlign': 'center', 'padding': '20px', 'margin': '0'}),

    html.Div(id='live-update-text', style={'color': '#9ca3af', 'textAlign': 'center', 'marginBottom': '10px'}),

    dcc.Tabs(id='tabs', value='tab-gex-charm', style={'backgroundColor': '#1f2937'}, children=[
        dcc.Tab(label='📊 GEX & CHARM', value='tab-gex-charm',
                style={'backgroundColor': '#1f2937', 'color': '#9ca3af'},
                selected_style={'backgroundColor': '#374151', 'color': '#e5e7eb', 'fontWeight': 'bold'}),
        dcc.Tab(label='🌡️ Volatilidad IV', value='tab-volatility',
                style={'backgroundColor': '#1f2937', 'color': '#9ca3af'},
                selected_style={'backgroundColor': '#374151', 'color': '#e5e7eb', 'fontWeight': 'bold'}),
    ]),

    html.Div(id='tabs-content'),

    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    )
])

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-gex-charm':
        return html.Div([
            dcc.Graph(id='gex-profile-graph', style={'height': '80vh'}),
            dcc.Graph(id='charm-profile-graph', style={'height': '80vh', 'marginTop': '10px'}),
        ])
    elif tab == 'tab-volatility':
        return html.Div([
            dcc.Graph(id='volatility-heatmap', style={'height': '90vh'}),
        ])

@app.callback(
    [Output('gex-profile-graph', 'figure'),
     Output('charm-profile-graph', 'figure'),
     Output('volatility-heatmap', 'figure'),
     Output('live-update-text', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_all_graphs(n):
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ========== ACTUALIZANDO DASHBOARD UNIFICADO ==========")

        print("📊 Cargando datos GEX/CHARM...")
        spot, df_greeks = build_greeks_df(ticker=TICKER, max_exp=8, r=0.05, q=0.015, min_oi=100)
        fig_gex = plot_trace_style_plotly(df_greeks, spot, ticker=TICKER, value="GEX", exp="nearest", zoom_pct=0.02)
        fig_charm = plot_trace_style_plotly(df_greeks, spot, ticker=TICKER, value="CHARM", exp="nearest", zoom_pct=0.02)
        print("✅ GEX/CHARM completado")

        print("🌡️ Cargando datos de Volatilidad...")
        vix_hist, current_vix = get_vix_data()
        vix_zscore = z_last(vix_hist.iloc[:-1].values, current_vix)
        pivot_data, spot_price = load_and_build_matrix(ticker=TICKER)
        fig_vol = create_heatmap_figure(pivot_data, spot_price, vix_zscore, TICKER)
        print("✅ Volatilidad completado")

        update_time = f"✅ Última Actualización: {datetime.now().strftime('%H:%M:%S')} | Spot: ${spot:.2f} | VIX: {current_vix:.2f} (Z: {vix_zscore:+.2f})"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ========== ACTUALIZACIÓN COMPLETA ==========\n")

        return fig_gex, fig_charm, fig_vol, update_time

    except Exception as e:
        print(f"❌ Error durante la actualización: {e}")
        import traceback
        traceback.print_exc()

        error_fig = go.Figure().update_layout(
            template="plotly_dark",
            title_text=f"Error al obtener datos. Reintentando...",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        return error_fig, error_fig, error_fig, f"❌ Error: {e}"

if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════╗
║                 QUANT TERMINAL - UNIFIED DASHBOARD             ║
║                                                                ║
║  🚀 Dashboard iniciado en: http://localhost:8050               ║
║  📊 Pestañas disponibles:                                      ║
║     - GEX & CHARM: Exposición gamma y delta decay             ║
║     - Volatilidad IV: Superficie de volatilidad implícita     ║
║                                                                ║
║  ⏱️  Actualización automática cada 5 minutos                   ║
║  🎯 Ticker: SPY                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, port=8050)
