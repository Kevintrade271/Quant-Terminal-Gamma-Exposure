#!/usr/bin/env python3
# gex_qtp_loop_pdf_v4_mirror.py
# ------------------------------------------------------------
# QTP — GEX intradía con yfinance + LOOP + PDF RN
# - Barras “espejo”: Calls (verde, derecha), Puts (rojo, izquierda)
# - Perfil de Gamma Neta (continuous)
# - Superposición opcional de volumen por strike (eje superior)
# - Etiquetas p1/p2/p3 por lado y en el perfil
# - Modo ODTE: --odte_only (y opcional --odte_pdf)
# - LEYENDAS operativas en todos los gráficos
# - PDF coloreado estilo “púrpura con borde azul”
# ------------------------------------------------------------

import argparse, math, time, datetime as dt
import numpy as np, pandas as pd, matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Instala yfinance: pip install yfinance") from e

# =============== Default params ===============
DEFAULTS = dict(
    ticker='GLD',
    semanales=2,
    mensuales=3,
    xticks=20,
    dealer='spotgamma',     # 'spotgamma' (calls -, puts +) | 'short_both' (ambos -)
    oi_min=10,
    max_moneyness=0.25,     # ±25% del spot
    iv_floor=0.03, iv_cap=3.0,
    lambda_T=3.0,           # peso_T = exp(-λT)
    grid_points=200,
    pdf_min_days=2, pdf_max_days=21,
)

COLORS = {
    "line": "#d1d5db",
    "pos_fill": "#16a34a",
    "neg_fill": "#b91c1c",
    "spot": "#93c5fd",
    "flip": "#22c55e",
    "call_wall": "#60a5fa",
    "put_wall": "#ef4444",
    "Slip Risk": "#34d399",
    "Gamma Pin": "#f59e0b",
    "Exhaustion": "#fb923c",
    "Convex Hotspot": "#a78bfa",
    "spot_pdf": "#94a3b8",
    "bubbles": "#94f7ef",
    # Barras espejo
    "call_bar": "#16a34a",  # verde
    "put_bar":  "#b91c1c",  # rojo
    # PDF estilo “púrpura con borde azul”
    "pdf_fill": "#a855f7",  # purple-500
    "pdf_edge": "#60a5fa",  # blue-400
}

# =============== Utils ===============
def style_dark(): plt.style.use("dark_background")
def normal_pdf(x): return np.exp(-0.5*x*x)/math.sqrt(2.0*math.pi)

def rf_from_irx(default=0.045):
    try:
        return float(yf.Ticker("^IRX").history(period="5d").Close.dropna().iloc[-1])/100.0
    except Exception:
        return float(default)

def yearfrac_to_close(venc):
    close_dt = dt.datetime.strptime(venc, "%Y-%m-%d") + dt.timedelta(hours=20)  # ~16:00 ET
    now = dt.datetime.utcnow()
    T = (close_dt - now).total_seconds()/(365*24*3600)
    return max(T, 1/365)

def dealer_sign_vector(tipo_array, convention):
    if convention=='spotgamma':  # calls -, puts +
        return np.where(tipo_array=='call', -1.0, +1.0)
    return np.full_like(tipo_array, -1.0, dtype=float)  # ambos -

def resolve_spot_opt(ticker_in: str):
    t = (ticker_in or "").upper()
    presets = {
        "SPX":   ("^GSPC", "SPY",  "SPX (SPY opts)"),
        "^GSPC": ("^GSPC", "SPY",  "SPX (SPY opts)"),
        "SPY":   ("SPY",   "SPY",  "SPY"),
        "NDX":   ("^NDX",  "QQQ",  "NDX (QQQ opts)"),
        "^NDX":  ("^NDX",  "QQQ",  "NDX (QQQ opts)"),
        "QQQ":   ("QQQ",   "QQQ",  "QQQ"),
        "GLD":   ("GLD",   "GLD",  "GLD"),
        "XAU":   ("GLD",   "GLD",  "Gold (GLD opts)"),
        "NAS100":"^NDX",
        "US500": "^GSPC",
    }
    spot, opt, label = presets.get(t, (t, t, t))
    if spot in ("^GSPC", "^NDX") and opt == spot:
        opt = "SPY" if spot=="^GSPC" else "QQQ"
        label = f"{spot} ({opt} opts)"
    return spot, opt, label

# =============== Data fetchers ===============
def fetch_spot(spot_ticker):
    t = yf.Ticker(spot_ticker)
    h = t.history(period="5d", interval="1m")
    if not h.empty: return float(h["Close"].dropna().iloc[-1])
    h = t.history(period="1d")
    if h.empty: raise RuntimeError(f"Sin precio para {spot_ticker}")
    return float(h["Close"].iloc[-1])

def fetch_option_chain(opt_ticker, semanales, mensuales, verbose=True):
    t = yf.Ticker(opt_ticker)
    expir = list(t.options) if getattr(t, "options", None) else []
    if not expir:
        raise RuntimeError(f"{opt_ticker} no tiene expiraciones en Yahoo.")

    # OPEX (tercer viernes)
    def opex_list(y0, yrs=2):
        out = []
        for y in range(y0, y0 + yrs):
            for m in range(1, 13):
                cand = [dt.date(y, m, d) for d in range(15, 22) if dt.date(y, m, d).weekday() == 4]
                if cand: out.append(cand[0].strftime("%Y-%m-%d"))
        return out

    expir_list = list(expir)
    weekly = list(expir_list[:semanales])
    opex_candidates = [d for d in opex_list(dt.date.today().year) if d in expir_list]
    vencs = sorted(list(set(weekly + opex_candidates[:mensuales])))

    frames = []
    def try_add(v):
        for k in range(3):
            try:
                ch = t.option_chain(v)
                calls, puts = ch.calls.copy(), ch.puts.copy()
                if not calls.empty and not puts.empty:
                    calls["tipo"], puts["tipo"] = "call", "put"
                    df = pd.concat([calls, puts], ignore_index=True)
                    df["vencimiento"] = v
                    frames.append(df); return
            except Exception:
                pass
            time.sleep(0.7 * (k + 1))

    for v in vencs: try_add(v)
    if not frames:
        extra = expir_list[:10]
        if verbose:
            print("[WARN] Cadena vacía con selección inicial. Amplío expiraciones:", extra)
        for v in extra: try_add(v)
    if not frames: raise RuntimeError(f"No se pudieron descargar opciones de {opt_ticker}.")
    return pd.concat(frames, ignore_index=True), vencs

def clean_chain(chain, spot, *, iv_floor, iv_cap, oi_min, max_moneyness):
    df = chain.copy()
    for c in ["impliedVolatility","strike","openInterest","tipo","vencimiento"]:
        if c not in df.columns: raise RuntimeError(f"Falta columna: {c}")
    df["tiempo_en_anios"] = df["vencimiento"].apply(yearfrac_to_close).astype(float)
    iv = pd.to_numeric(df["impliedVolatility"], errors="coerce").ffill().bfill()
    df["iv"] = np.clip(iv.values.astype(float), iv_floor, iv_cap)
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(float)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(float)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").astype(float)
    df["tipo"] = df["tipo"].astype(str)
    lo, hi = spot*(1-max_moneyness), spot*(1+max_moneyness)
    df = df[(df["strike"]>=lo) & (df["strike"]<=hi)]
    df = df[(df["openInterest"]>=oi_min) & (df["tiempo_en_anios"]>0)]
    df.reset_index(drop=True, inplace=True)
    return df

# =============== GEX core ===============
def gex_nocional_vectorizado(S, df, r, dealer, lambda_T):
    K  = df["strike"].values.astype(float)
    T  = df["tiempo_en_anios"].values.astype(float)
    iv = df["iv"].values.astype(float)
    oi = df["openInterest"].values.astype(float)
    typ= df["tipo"].values
    with np.errstate(all="ignore"):
        d1 = (np.log(S/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
    gamma = normal_pdf(d1)/(S*iv*np.sqrt(T))
    sign  = dealer_sign_vector(typ, dealer)
    wT    = np.exp(-lambda_T*T)
    gex   = gamma * (S**2) * 0.01 * 100.0 * oi * sign * wT
    return float(np.nansum(gex))

def gex_components_at_spot(df, spot, r, dealer, lambda_T):
    K  = df["strike"].values.astype(float)
    T  = df["tiempo_en_anios"].values.astype(float)
    iv = df["iv"].values.astype(float)
    oi = df["openInterest"].values.astype(float)
    typ= df["tipo"].values
    with np.errstate(all="ignore"):
        d1 = (np.log(spot/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T))
    gamma = normal_pdf(d1)/(spot*iv*np.sqrt(T))
    sign  = dealer_sign_vector(typ, dealer)
    wT    = np.exp(-lambda_T*T)
    gex_v = gamma * (spot**2) * 0.01 * 100.0 * oi * sign * wT
    tmp = pd.DataFrame({"strike":K,"tipo":typ,"gex":gex_v})
    total = tmp.groupby("strike")["gex"].sum()/1e9
    calls = tmp[tmp["tipo"]=="call"].groupby("strike")["gex"].sum()/1e9
    puts  = tmp[tmp["tipo"]=="put"].groupby("strike")["gex"].sum()/1e9
    return total.sort_index(), calls.sort_index(), puts.sort_index()

def gamma_flip_from_curve(x, y):
    s = np.sign(y); idx = np.where(np.diff(s)!=0)[0]
    if len(idx)==0: return None
    i = idx[0]
    return float(x[i] - y[i]*(x[i+1]-x[i])/(y[i+1]-y[i]))

# Volumen por strike
def volume_components_by_strike(df):
    tmp = df.groupby(["strike","tipo"])["volume"].sum().unstack(fill_value=0)
    calls = tmp.get("call", pd.Series(dtype=float)).sort_index()
    puts  = tmp.get("put",  pd.Series(dtype=float)).sort_index()
    return calls, puts

# =============== RN-PDF (Breeden–Litzenberger) ===============
def choose_expiration(opts, min_days=2, max_days=21):
    today = dt.datetime.utcnow().date()
    best, best_dt = None, 10**9
    for e in opts:
        try:
            d = dt.datetime.strptime(e, "%Y-%m-%d").date()
            dd = (d - today).days
            if dd >= min_days and (dd <= max_days or best is None) and dd < best_dt:
                best, best_dt = e, dd
        except Exception:
            pass
    return best, best_dt

def load_chain_for_pdf(opt_ticker, min_days, max_days):
    t = yf.Ticker(opt_ticker)
    exps = t.options or []
    if not exps: raise RuntimeError(f"{opt_ticker} sin expiraciones PDF.")
    exp, dte = choose_expiration(exps, min_days, max_days)
    if exp is None: exp, dte = exps[0], 7
    ch = t.option_chain(exp); calls, puts = ch.calls.copy(), ch.puts.copy()
    for df in (calls, puts):
        df.rename(columns={"openInterest":"oi"}, inplace=True)
        for c in ["bid","ask","lastPrice","strike","oi"]:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)
        mid = (df["bid"] + df["ask"])/2.0
        df["mid"] = np.where(mid>0, mid, df["lastPrice"]).astype(float)
    return calls, puts, exp, dte

def estimate_forward(calls, puts, r, T):
    m = pd.merge(calls[["strike","mid"]], puts[["strike","mid"]], on="strike", suffixes=("_c","_p"))
    if m.empty: raise RuntimeError("Sin strikes comunes para forward.")
    m["cp_spread"] = m["mid_c"] - m["mid_p"]
    Kmin, Kmax = m["strike"].quantile(0.15), m["strike"].quantile(0.85)
    band = m[(m["strike"]>=Kmin) & (m["strike"]<=Kmax)].copy()
    band["F_i"] = band["strike"] + math.exp(r*T)*band["cp_spread"]
    return float(band["F_i"].median())

def build_call_curve(calls, puts, F, r, T):
    Ks = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
    grid = pd.DataFrame({"strike": Ks})
    c_map = dict(zip(calls["strike"], calls["mid"]))
    p_map = dict(zip(puts["strike"],  puts["mid"]))
    mids_c=[]
    for K in grid["strike"]:
        if K>=F and K in c_map: mids_c.append(float(c_map[K]))
        elif K<F and K in p_map: mids_c.append(float(p_map[K]) + math.exp(-r*T)*(F-K))
        else:
            mc, mp = c_map.get(K, np.nan), p_map.get(K, np.nan)
            mids_c.append(float(mc) if not np.isnan(mc) else (float(mp)+math.exp(-r*T)*(F-K) if not np.isnan(mp) else np.nan))
    grid["call_mid"]=mids_c
    grid = grid.dropna().sort_values("strike").reset_index(drop=True)
    if len(grid)<5: raise RuntimeError("Curva de calls insuficiente.")
    dK = float(np.median(np.diff(grid["strike"].values)))
    K_grid = np.arange(grid["strike"].min(), grid["strike"].max()+dK, dK)
    call_interp = np.interp(K_grid, grid["strike"].values, grid["call_mid"].values)
    out = pd.DataFrame({"strike":K_grid, "call_mid":call_interp}); out["dK"]=dK
    return out

def pdf_from_calls(call_curve, r, T):
    K = call_curve["strike"].values; C = call_curve["call_mid"].values; dK = float(call_curve["dK"].iloc[0])
    second = np.zeros_like(C); second[1:-1] = (C[:-2]-2*C[1:-1]+C[2:])/(dK*dK)
    f = np.exp(r*T)*second; f = np.clip(f, 0.0, None)
    integ = np.sum(f)*dK
    if integ>0: f /= integ
    return pd.DataFrame({"strike":K, "pdf":f, "dK":dK})

def targets_from_gex(total_series, spot):
    gabs = total_series.abs()
    win = gabs[(gabs.index>=spot*0.98)&(gabs.index<=spot*1.02)]
    gamma_pin = float(win.idxmax() if not win.empty else gabs.idxmax())
    put_wall  = float(total_series.idxmax())
    call_wall = float(total_series.idxmin())
    g = gabs.sort_index(); curv = (g.shift(-1) - 2*g + g.shift(1)).abs()
    curv_above = curv[curv.index>spot]
    convex_hot = float(curv_above.idxmax()) if not curv_above.empty else float(g.idxmax())
    return {"Slip Risk":put_wall, "Gamma Pin":gamma_pin, "Exhaustion":call_wall, "Convex Hotspot":convex_hot}

def probs_from_pdf(pdf_df, targets, spot):
    dK = float(pdf_df["dK"].iloc[0]); w = max(2.0*dK, 0.003*spot)
    K = pdf_df["strike"].values; f = pdf_df["pdf"].values
    probs = {name: float(np.sum(f[(K>=k-w)&(K<=k+w)])*dK) for name,k in targets.items()}
    s = sum(probs.values())
    if s>0: probs = {k: v/s for k,v in probs.items()}
    return probs

# =============== Helpers de plotting y leyendas ===============
def _union_index(a, b):
    idx = sorted(set(a.index).union(set(b.index)))
    return (a.reindex(idx).fillna(0.0), b.reindex(idx).fillna(0.0), idx)

def _subset_window(series, spot, pct=0.10, min_n=15):
    lo, hi = spot*(1-pct), spot*(1+pct)
    sub = series[(series.index>=lo)&(series.index<=hi)]
    if len(sub) < min_n: sub = series
    return sub

def _nearest_index(strikes, level):
    arr = np.array(strikes, dtype=float)
    return int(np.argmin(np.abs(arr - float(level))))

def _set_y_ticks(ax, strikes, values_a, values_b=None, max_labels=20):
    mag = np.abs(values_a)
    if values_b is not None: mag = mag + np.abs(values_b)
    order = np.argsort(mag)[::-1][:min(max_labels, len(strikes))]
    yt = sorted(order.tolist())
    ax.set_yticks(yt); ax.set_yticklabels([int(round(strikes[i])) for i in yt], fontsize=9)

def _top_k_indices(values: np.ndarray, k: int, *, key_abs=False):
    values = np.asarray(values)
    if values.size == 0: return np.array([], dtype=int)
    order = np.argsort(np.abs(values) if key_abs else values)[::-1]
    return order[:min(k, values.size)]

def _annotate_bar_labels(ax, xs, ys, labels, *, align='right', dx=0.02):
    xlim = ax.get_xlim(); span = xlim[1] - xlim[0]
    for x, y, lab in zip(xs, ys, labels):
        off = dx * span * (1 if align=='right' else -1)
        ax.text(x + off, y, lab, va="center", ha=("left" if align=='right' else "right"),
                fontsize=10, fontweight="bold")

def add_legend_note(ax, text, *, loc="lower right", fontsize=9, alpha=0.22):
    anchors = {
        "lower right": (0.99, 0.01, "right", "bottom"),
        "lower left":  (0.01, 0.01, "left",  "bottom"),
        "upper right": (0.99, 0.99, "right", "top"),
        "upper left":  (0.01, 0.99, "left",  "top"),
    }
    x, y, ha, va = anchors.get(loc, (0.99, 0.01, "right", "bottom"))
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=alpha, edgecolor="none"),
            color="white")

# =============== Plots ===============
def plot_gamma_profile(label, spot, strikes, profile, flip, call_wall, put_wall, show=True, save_path=None):
    style_dark()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.fill_between(strikes, profile, 0, where=profile>=0, facecolor=COLORS["pos_fill"], alpha=0.45, zorder=1)
    ax.fill_between(strikes, profile, 0, where=profile<0,  facecolor=COLORS["neg_fill"], alpha=0.45, zorder=1)
    ax.plot(strikes, profile, color=COLORS["line"], linewidth=1.4, zorder=2)
    ax.axhline(0, color="#fff", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axvline(spot, color=COLORS["spot"], linestyle=":", linewidth=1.2, label=f"Spot: {int(round(spot))}")
    if flip is not None: ax.axvline(flip, color=COLORS["flip"], linestyle="-", linewidth=1.5, label=f"Gamma Flip: {int(round(flip))}")
    ax.axvline(call_wall, color=COLORS["call_wall"], linestyle="--", linewidth=1.2, label=f"Call Wall: {int(round(call_wall))}")
    ax.axvline(put_wall,  color=COLORS["put_wall"],  linestyle="--", linewidth=1.2, label=f"Put Wall: {int(round(put_wall))}")
    ax.set_title(f"{label} — Gamma Profile (GEX $B / 1%)"); ax.set_xlabel("Strike"); ax.set_ylabel("GEX ($B / 1%)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4); ax.legend()
    lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])); ax.set_ylim(-lim, lim)

    prof = np.asarray(profile, dtype=float); strikes_arr = np.asarray(strikes, dtype=float)
    pos_mask = prof > 0
    if np.any(pos_mask):
        vals_pos = prof[pos_mask]; idx_pos = _top_k_indices(vals_pos, 3, key_abs=False)
        for i, (xv, yv) in enumerate(zip(strikes_arr[pos_mask][idx_pos], vals_pos[idx_pos]), start=1):
            ax.annotate(f"p{i}", xy=(xv, yv), xytext=(0, 10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color=COLORS["pos_fill"])
    neg_mask = prof < 0
    if np.any(neg_mask):
        vals_neg = prof[neg_mask]; idx_neg = _top_k_indices(vals_neg, 3, key_abs=True)
        for i, (xv, yv) in enumerate(zip(strikes_arr[neg_mask][idx_neg], vals_neg[idx_neg]), start=1):
            ax.annotate(f"p{i}", xy=(xv, yv), xytext=(0, -12), textcoords="offset points",
                        ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS["neg_fill"])

    legend_text = (
        "• Gamma>0 (verde): sesgo mean-revert\n"
        "• Gamma<0 (rojo): sesgo trend/aceleración\n"
        "• Flip: cambio de régimen (bajo=acel., sobre=freno)\n"
        "• Call/Put Wall: techo/soporte probabilísticos\n"
        "• p1–p3(+): picos amortiguadores\n"
        "• p1–p3(−): valles de fragilidad"
    )
    add_legend_note(ax, legend_text, loc="lower right", fontsize=9, alpha=0.22)

    if save_path: plt.savefig(save_path, dpi=140, bbox_inches="tight")
    if show: plt.show(); plt.close(fig)

def plot_gex_total_mirror(label, spot, total_series, xticks_n, show=True, save_path=None):
    style_dark()
    sub = _subset_window(total_series, spot, pct=0.10)
    strikes = list(sub.index.astype(float))
    vals = sub.values.astype(float)
    y = np.arange(len(strikes))
    colors = [COLORS["pos_fill"] if v>0 else COLORS["neg_fill"] for v in vals]

    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.axvline(0, color="#999", linestyle="--", linewidth=1.0)
    spot_idx = _nearest_index(strikes, spot)
    ax.axhline(spot_idx, color=COLORS["spot"], linestyle=":", linewidth=1.2, label=f"Spot {int(round(spot))}")

    _set_y_ticks(ax, strikes, vals, None, max_labels=xticks_n)
    ax.invert_yaxis()
    ax.set_xlabel("GEX ($B / 1%)"); ax.set_ylabel("Strike")
    ax.set_title(f"{label} — GEX Total por strike (formato espejo)")
    ax.grid(axis="x", linestyle=":", alpha=0.35); ax.legend()

    lim = max(np.max(np.abs(vals)), 1e-6)
    ax.set_xlim(-lim*1.1, lim*1.1)

    legend_text = (
        "• Barras verdes: freno (gamma+)\n"
        "• Barras rojas: aceleración (gamma−)\n"
        "• Spot: referencia actual\n"
        "• Bloques densos ⇒ rango operativo\n"
        "• Extremos del bloque: entradas contrarias\n"
        "• Stop: 0.6–1.0×ATR según régimen"
    )
    add_legend_note(ax, legend_text, loc="upper right", fontsize=9, alpha=0.22)

    if save_path: plt.savefig(save_path, dpi=140, bbox_inches="tight")
    if show: plt.show(); plt.close(fig)

def plot_calls_puts_mirror(label, spot, calls_s, puts_s, *, 
                           call_wall=None, put_wall=None,
                           xticks_n=20, title_suffix="",
                           vol_calls_s=None, vol_puts_s=None,
                           show=True, save_path=None):
    style_dark()
    calls_s, puts_s, _ = _union_index(calls_s, puts_s)
    c_sub = _subset_window(calls_s, spot, pct=0.10)
    p_sub = puts_s.reindex(c_sub.index).fillna(0.0)
    strikes = list(c_sub.index.astype(float))

    calls_vals = np.abs(c_sub.values.astype(float))   # derecha
    puts_vals  = -np.abs(p_sub.values.astype(float))  # izquierda
    y = np.arange(len(strikes))

    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(y, calls_vals, color=COLORS["call_bar"], alpha=0.9, label="Calls (|GEX|)")
    ax.barh(y, puts_vals,  color=COLORS["put_bar"],  alpha=0.9, label="Puts  (|GEX|)")

    ax.axvline(0, color="#999", linestyle="--", linewidth=1.0)
    spot_idx = _nearest_index(strikes, spot)
    ax.axhline(spot_idx, color=COLORS["spot"], linestyle=":", linewidth=1.2, label=f"Spot {int(round(spot))}")

    if call_wall is not None:
        ax.axhline(_nearest_index(strikes, call_wall), color=COLORS["call_wall"], linestyle="--", linewidth=1.0, label=f"Call Wall {int(round(call_wall))}")
    if put_wall is not None:
        ax.axhline(_nearest_index(strikes, put_wall), color=COLORS["put_wall"], linestyle="--", linewidth=1.0, label=f"Put Wall {int(round(put_wall))}")

    _set_y_ticks(ax, strikes, calls_vals, np.abs(puts_vals), max_labels=xticks_n)
    ax.invert_yaxis()
    ax.set_xlabel("|GEX| ($B / 1%)  ← Puts | Calls →")
    ax.set_ylabel("Strike")
    ax.set_title(f"{label} — Calls vs Puts por strike (espejo) {('— '+title_suffix) if title_suffix else ''}")
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    lim = max(np.max(np.abs(calls_vals)), np.max(np.abs(puts_vals)), 1e-6)
    ax.set_xlim(-lim*1.1, lim*1.1)

    idx_top_calls = _top_k_indices(calls_vals, 3, key_abs=False)
    _annotate_bar_labels(ax, calls_vals[idx_top_calls], y[idx_top_calls], [f"p{i+1}" for i in range(len(idx_top_calls))],
                         align='right', dx=0.02)
    idx_top_puts = _top_k_indices(puts_vals, 3, key_abs=True)
    _annotate_bar_labels(ax, puts_vals[idx_top_puts], y[idx_top_puts], [f"p{i+1}" for i in range(len(idx_top_puts))],
                         align='left', dx=0.02)

    # Volumen overlay
    if (vol_calls_s is not None) or (vol_puts_s is not None):
        vc_series = vol_calls_s if isinstance(vol_calls_s, pd.Series) else pd.Series(dtype=float)
        vp_series = vol_puts_s  if isinstance(vol_puts_s,  pd.Series) else pd.Series(dtype=float)
        vc = vc_series.reindex(c_sub.index).fillna(0.0).to_numpy(dtype=float)
        vp = vp_series.reindex(c_sub.index).fillna(0.0).to_numpy(dtype=float)
        try:
            vmax = max(vc.max(initial=0.0), vp.max(initial=0.0), 1.0)
        except TypeError:
            vmax = max((vc.max() if vc.size else 0.0), (vp.max() if vp.size else 0.0), 1.0)

        ax2 = ax.twiny(); ax2.set_xlim(-vmax, vmax); ax2.set_xlabel("Volumen (contratos)  ← Puts | Calls →")
        scale = (lim*1.0) / vmax
        ax.fill_betweenx(y, 0,  scale * vc, alpha=0.25, color="#60a5fa", label="Call Volume")
        ax.fill_betweenx(y, 0, -scale * vp, alpha=0.25, color="#fb923c", label="Put Volume")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=2, fontsize=9)

    legend_text = (
        "• Calls → derecha (resistencias prob.)\n"
        "• Puts  ← izquierda (soportes prob.)\n"
        "• p1–p3: top-3 por |GEX| en cada lado\n"
        "• Volumen (áreas): refuerza el nivel\n"
        "• En gamma−: rupturas aceleran\n"
        "• En gamma+: extremos tienden a revertir"
    )
    add_legend_note(ax, legend_text, loc="upper right", fontsize=9, alpha=0.22)

    if save_path: plt.savefig(save_path, dpi=140, bbox_inches="tight")
    if show: plt.show(); plt.close(fig)

# =============== PDF Plots (estilo púrpura + borde azul) ===============
def _jitter(xs, min_sep=0.6):
    xs=xs.astype(float).copy()
    for i in range(len(xs)):
        for j in range(i):
            if abs(xs[i]-xs[j])<min_sep:
                xs[i]+= (min_sep if (i-j)%2==0 else -min_sep)
    return xs

def plot_pdf(pdf_df, spot, targets, probs, title, show=True, save_path=None):
    style_dark()
    fig, ax = plt.subplots(figsize=(12,6))
    K = pdf_df["strike"].values
    f = pdf_df["pdf"].values
    dK = float(pdf_df["dK"].iloc[0])

    ax.bar(K, f, width=dK, align="center", color=COLORS["pdf_fill"], edgecolor=COLORS["pdf_edge"], linewidth=0.8, alpha=0.75)

    ax.axvline(spot, color=COLORS["spot_pdf"], linestyle=":", linewidth=1.1, label=f"Spot: {int(round(spot))}")
    for name,K0 in targets.items():
        ax.axvline(K0, color=COLORS[name], linestyle="--", linewidth=1.1, label=f"{name}: {int(round(K0))}")

    ax.set_title(f"Risk-Neutral PDF — {title}")
    ax.set_xlabel("Strike"); ax.set_ylabel("Probabilidad por 1 USD")
    ax.grid(True, alpha=0.35)
    ax.legend(ncol=2, fontsize=9)

    legend_text = (
        "• PDF: distribución implícita a corto plazo\n"
        "• Pin: prob. de rango/lateral\n"
        "• Slip: prob. de barrida bajista\n"
        "• Exhaustion: techo probable\n"
        "• Convex: acel. alcista probable\n"
        "• Combinar con p1–p3 y Flip"
    )
    add_legend_note(ax, legend_text, loc="upper right", fontsize=9, alpha=0.22)

    if save_path: plt.savefig(save_path, dpi=140, bbox_inches="tight")
    if show: plt.show(); plt.close(fig)

def plot_pdf_bubbles(probs, targets, title, spot, show=True, save_path=None):
    style_dark()
    names=list(targets.keys()); x0=np.array([targets[n] for n in names],float); y=np.array([probs.get(n,0) for n in names],float)
    x=_jitter(x0, min_sep=max(0.4,0.003*spot)); sizes=(y/(y.max() if y.max()>0 else 1))*3500
    fig,ax=plt.subplots(figsize=(12,6))
    ax.scatter(x,y,s=sizes,alpha=0.75,edgecolors="none",c=COLORS["pdf_fill"])
    for xi,yi,name,kref in zip(x,y,names,x0):
        ax.text(xi,yi,f"{int(round(yi*100))}%",ha="center",va="center",fontsize=10)
        ax.annotate(name,xy=(xi,yi),xytext=(0,12),textcoords="offset points",ha="center",fontsize=10)
        ax.axvline(kref,color=COLORS[name],linestyle="--",linewidth=1.0)
    ax.axvline(spot,color=COLORS["spot_pdf"],linestyle=":",linewidth=1.1,label="Spot")
    ax.set_title(f"{title} — Probabilidades por objetivo (PDF-integradas)")
    ax.set_xlabel("Strike"); ax.set_ylabel("Probability")
    ax.set_ylim(0.0, max(0.5, y.max()+0.1)); ax.grid(True, alpha=0.35)
    ax.legend().set_visible(False)

    legend_text = (
        "• Tamaño ∝ prob. implícita por target\n"
        "• Pin domina: sesgo de rango\n"
        "• Slip/Convex dominan: sesgo de breakout\n"
        "• Usar con p1–p3 y Walls\n"
        "• Confirmar con VWAP/POC"
    )
    add_legend_note(ax, legend_text, loc="upper right", fontsize=9, alpha=0.22)

    if save_path: plt.savefig(save_path, dpi=140, bbox_inches="tight")
    if show: plt.show(); plt.close(fig)

# =============== ODTE helper ===============
def select_odte_slice(df):
    today = dt.datetime.utcnow().date()
    df = df.copy()
    df["DTE"] = df["vencimiento"].apply(lambda v: (dt.datetime.strptime(v, "%Y-%m-%d").date() - today).days)
    candid = df[df["DTE"]==0]
    if candid.empty: candid = df[df["DTE"]>=0]
    if candid.empty:
        min_abs = df.iloc[(df["DTE"].abs()).argsort().values]
        min_dte = min_abs["DTE"].iloc[0]
        candid = df[df["DTE"]==min_dte]
    v0 = candid["vencimiento"].iloc[0]
    return df[df["vencimiento"]==v0], v0

# =============== Run once / Loop ===============
def run_once(args, iteration=0):
    spot_tkr, opt_tkr, label = resolve_spot_opt(args.ticker)
    spot = fetch_spot(spot_tkr)
    r = rf_from_irx()

    chain_raw, _ = fetch_option_chain(opt_tkr, args.semanales, args.mensuales)
    df_all = clean_chain(chain_raw, spot, iv_floor=args.iv_floor, iv_cap=args.iv_cap,
                         oi_min=args.oi_min, max_moneyness=args.max_moneyness)

    odte_df, odte_venc = select_odte_slice(df_all)
    if args.odte_only:
        df_base = odte_df
        label_suffix = f" — ODTE {odte_venc}"
    else:
        df_base = df_all
        label_suffix = ""

    total_s, calls_s, puts_s = gex_components_at_spot(df_base, spot, r, args.dealer, args.lambda_T)
    call_wall = float((calls_s.idxmin() if not calls_s.empty else total_s.idxmin()))
    put_wall  = float((puts_s.idxmax()  if not puts_s.empty  else total_s.idxmax()))

    lo, hi = spot*0.85, spot*1.15
    grid = np.linspace(lo, hi, args.grid_points)
    profile = np.array([gex_nocional_vectorizado(S, df_base, r, args.dealer, args.lambda_T) for S in grid]) / 1e9
    flip = gamma_flip_from_curve(grid, profile)

    targets = targets_from_gex(total_s, spot)
    pdf_df, probs = None, {}
    try:
        if args.odte_only and args.odte_pdf:
            t = yf.Ticker(opt_tkr)
            ch = t.option_chain(odte_venc)
            calls_pdf, puts_pdf = ch.calls.copy(), ch.puts.copy()
            for dfp in (calls_pdf, puts_pdf):
                dfp.rename(columns={"openInterest":"oi"}, inplace=True)
                for c in ["bid","ask","lastPrice","strike","oi"]:
                    dfp[c] = pd.to_numeric(dfp.get(c, 0), errors="coerce").fillna(0.0)
                mid = (dfp["bid"] + dfp["ask"]) / 2.0
                dfp["mid"] = np.where(mid>0, mid, dfp["lastPrice"]).astype(float)
            T_pdf = max((dt.datetime.strptime(odte_venc, "%Y-%m-%d") - dt.datetime.utcnow()).days/365.0, 1/365.0)
            F_pdf = estimate_forward(calls_pdf, puts_pdf, r, T_pdf)
            curve_pdf = build_call_curve(calls_pdf, puts_pdf, F_pdf, r, T_pdf)
            pdf_df = pdf_from_calls(curve_pdf, r, T_pdf)
            probs  = probs_from_pdf(pdf_df, targets, spot)
        else:
            calls_p, puts_p, exp_p, dte_p = load_chain_for_pdf(opt_tkr, args.pdf_min_days, args.pdf_max_days)
            T_p = max(dte_p/365.0, 1/365.0)
            F_p = estimate_forward(calls_p, puts_p, r, T_p)
            curve_p = build_call_curve(calls_p, puts_p, F_p, r, T_p)
            pdf_df = pdf_from_calls(curve_p, r, T_p)
            probs  = probs_from_pdf(pdf_df, targets, spot)
    except Exception:
        pdf_df, probs = None, {}

    vol_calls_base, vol_puts_base = volume_components_by_strike(df_base)

    prefix = f"{args.out_prefix}_{iteration:04d}" if args.out_prefix else None
    def p(n): return f"{prefix}_{n}.png" if prefix else None

    plot_gamma_profile(label + label_suffix, spot, grid, profile, flip, call_wall, put_wall,
                       show=not args.no_show, save_path=p("gex_profile"))

    plot_calls_puts_mirror(label + label_suffix, spot, calls_s, puts_s,
                           call_wall=call_wall, put_wall=put_wall,
                           xticks_n=args.xticks,
                           title_suffix=("ODTE" if args.odte_only else "todas las expiraciones"),
                           vol_calls_s=vol_calls_base, vol_puts_s=vol_puts_base,
                           show=not args.no_show,
                           save_path=p("calls_puts_ODTE_espejo_VOL" if args.odte_only else "calls_puts_all_espejo_VOL"))

    if not args.odte_only:
        odte_total, odte_calls, odte_puts = gex_components_at_spot(odte_df, spot, r, args.dealer, args.lambda_T)
        odte_call_wall = float(odte_calls.idxmin()) if not odte_calls.empty else None
        odte_put_wall  = float(odte_puts.idxmax())  if not odte_puts.empty  else None
        vol_calls_odte, vol_puts_odte = volume_components_by_strike(odte_df)
        plot_calls_puts_mirror(label, spot, odte_calls, odte_puts,
                               call_wall=odte_call_wall, put_wall=odte_put_wall,
                               xticks_n=args.xticks,
                               title_suffix=f"ODTE {odte_venc}",
                               vol_calls_s=vol_calls_odte, vol_puts_s=vol_puts_odte,
                               show=not args.no_show, save_path=p("calls_puts_ODTE_espejo_VOL"))

    if pdf_df is not None:
        plot_pdf(pdf_df, spot, targets, probs, title=label + label_suffix,
                 show=not args.no_show, save_path=p("pdf_profile"))
        plot_pdf_bubbles(probs, targets, title=label + label_suffix, spot=spot,
                         show=not args.no_show, save_path=p("pdf_bubbles"))

    # Log
    print(f"\n[{label}{label_suffix}] Spot {spot:.2f} ({spot_tkr}) | Opt {opt_tkr}")
    if flip is not None: print(f"Gamma Flip: {int(round(flip))}")
    print(f"Put Wall: {int(round(put_wall))} | Call Wall: {int(round(call_wall))}")
    for k in ["Slip Risk","Gamma Pin","Exhaustion","Convex Hotspot"]:
        print(f"{k}: {int(round(targets[k]))} | PDF: {probs.get(k,0.0)*100:4.1f}%")

def main():
    ap=argparse.ArgumentParser(description="QTP GEX intradía — leyendas + PDF púrpura")
    ap.add_argument("--ticker", type=str, default=DEFAULTS["ticker"])
    ap.add_argument("--semanales", type=int, default=DEFAULTS["semanales"])
    ap.add_argument("--mensuales", type=int, default=DEFAULTS["mensuales"])
    ap.add_argument("--xticks", type=int, default=DEFAULTS["xticks"])
    ap.add_argument("--dealer", type=str, choices=["spotgamma","short_both"], default=DEFAULTS["dealer"])
    ap.add_argument("--oi_min", type=int, default=DEFAULTS["oi_min"])
    ap.add_argument("--max_moneyness", type=float, default=DEFAULTS["max_moneyness"])
    ap.add_argument("--iv_floor", type=float, default=DEFAULTS["iv_floor"])
    ap.add_argument("--iv_cap", type=float, default=DEFAULTS["iv_cap"])
    ap.add_argument("--lambda_T", type=float, default=DEFAULTS["lambda_T"])
    ap.add_argument("--grid_points", type=int, default=DEFAULTS["grid_points"])
    ap.add_argument("--loop", type=int, default=0, help="segundos entre refrescos (0=una corrida)")
    ap.add_argument("--out_prefix", type=str, default=None)
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--export_csv", type=str, default=None)
    ap.add_argument("--pdf_min_days", type=int, default=DEFAULTS["pdf_min_days"])
    ap.add_argument("--pdf_max_days", type=int, default=DEFAULTS["pdf_max_days"])
    ap.add_argument("--odte_only", action="store_true",
                    help="Usar solo la cadena 0DTE (o la más cercana >=0 DTE) para GEX y perfil.")
    ap.add_argument("--odte_pdf", action="store_true",
                    help="Intentar PDF RN con el mismo vencimiento ODTE (puede ser inestable).")
    args=ap.parse_args()

    if args.no_show and not args.out_prefix:
        print("[ADVERTENCIA] --no_show activo sin --out_prefix: no se guardarán PNGs.")

    if args.loop>0:
        i=0
        while True:
            try: run_once(args, iteration=i)
            except Exception as e: print(f"[ERROR] Iteración {i}: {e}")
            i+=1; time.sleep(args.loop)
    else:
        run_once(args, iteration=0)

if __name__=="__main__":
    main()
