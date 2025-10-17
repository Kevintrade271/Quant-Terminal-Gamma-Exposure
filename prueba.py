# ============================================================
# SPY – Gamma / Delta / Vanna (dark mode)
# ============================================================
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# --------------------------
# Parámetros
# --------------------------
N_EXPIRIES      = 4      # sumar 0DTE + próximas
MULTIPLIER      = 100
SPOT_SWEEP_PCT  = 0.05   # barrido ±5% para Delta/Vanna vs spot
WALL_QUANTILE   = 0.90   # percentil de gamma walls

# --------------------------
# Estética modo oscuro
# --------------------------
plt.rcParams.update({
    "figure.facecolor": "#0e1116",
    "axes.facecolor":   "#0e1116",
    "savefig.facecolor":"#0e1116",
    "axes.edgecolor":   "#2a2f3a",
    "axes.labelcolor":  "#e6edf3",
    "text.color":       "#e6edf3",
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "grid.color":       "#2a2f3a",
    "axes.grid":        True,
    "grid.linestyle":   ":",
    "grid.linewidth":   0.6,
})

# Colores (inspirados en la UI de la imagen)
C_GEX   = "#2dd4bf"  # turquesa
C_DELTA = "#f59e0b"  # naranja ámbar
C_VANNA = "#a78bfa"  # violeta
C_LINE  = "#94a3b8"  # gris claro
C_FLIP  = "#ffffff"  # blanco

# --------------------------
# BS utilidades (vector)
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
    dt_days = (expiry_ts - now_utc).total_seconds() / (24*3600)
    return max(dt_days / 365.0, 5e-4)

# --------------------------
# Descarga SPY + opciones
# --------------------------
tk = yf.Ticker("SPY")
S0 = float(tk.fast_info["last_price"])

# r (13w T-Bill ^IRX ~ %) y q (dividend yield aprox)
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
if not exps: raise SystemExit("Sin expiraciones disponibles.")
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
# Panel 1: GEX por strike (spot actual)
# --------------------------
opt["gamma"] = gamma_bs(S=S0, K=opt["strike"], T=opt["T"], r=r, q=q, sigma=opt["iv"])
opt["GEX"]   = opt["gamma"] * (S0**2) * opt["open_interest"] * MULTIPLIER
gex_by_strike = opt.groupby("strike")["GEX"].sum().sort_index()

# gamma flip (acumulado centrado) y walls
cum = (gex_by_strike - gex_by_strike.mean()).cumsum()
gamma_flip = cum.sub(0).abs().idxmin()
thr = gex_by_strike.quantile(WALL_QUANTILE)
gamma_walls = gex_by_strike[gex_by_strike >= thr].sort_values(ascending=False)

# --------------------------
# Paneles 2 y 3: Δ y Vanna agregadas vs spot
# --------------------------
S_grid = np.linspace(S0*(1-SPOT_SWEEP_PCT), S0*(1+SPOT_SWEEP_PCT), 75)

def book_delta_at(S):
    d = delta_bs(S=S, K=opt["strike"], T=opt["T"], r=r, q=q, sigma=opt["iv"], cp=opt["cp"])
    return np.sum(d * opt["open_interest"].values * MULTIPLIER)

def book_vanna_at(S):
    v = vanna_dDelta_dSigma(S=S, K=opt["strike"], T=opt["T"], r=r, q=q, sigma=opt["iv"], cp=opt["cp"])
    return np.sum(v * opt["open_interest"].values * MULTIPLIER)

delta_curve = np.array([book_delta_at(S) for S in S_grid])
vanna_curve = np.array([book_vanna_at(S) for S in S_grid])

# --------------------------
# Plot 3 paneles (dark)
# --------------------------
fig = plt.figure(figsize=(13,4.8))

# 1) GEX por strike
ax1 = plt.subplot(1,3,1)
ax1.plot(gex_by_strike.index, gex_by_strike.values, lw=1.8, color=C_GEX)
for k in gamma_walls.index[:3]:
    ax1.axvline(k, ls="--", lw=1.2, color=C_LINE, alpha=0.8)
ax1.axvline(gamma_flip, ls=":", lw=1.2, color=C_FLIP)
ax1.set_title("SPY — Gamma Model", color="#e6edf3")
ax1.set_xlabel("Strike"); ax1.set_ylabel("GEX (≈ γ·S²·OI·mult)")

# 2) Delta vs spot
ax2 = plt.subplot(1,3,2)
ax2.plot(S_grid, delta_curve, lw=1.8, color=C_DELTA)
ax2.axvline(S0, ls=":", lw=1.2, color=C_FLIP)
ax2.set_title("SPY — Delta Model (Δ agregado)", color="#e6edf3")
ax2.set_xlabel("Spot"); ax2.set_ylabel("Delta nocional (sum OI)")

# 3) Vanna vs spot
ax3 = plt.subplot(1,3,3)
ax3.plot(S_grid, vanna_curve, lw=1.8, color=C_VANNA)
ax3.axvline(S0, ls=":", lw=1.2, color=C_FLIP)
ax3.set_title("SPY — Vanna Model (∂Δ/∂σ agregado)", color="#e6edf3")
ax3.set_xlabel("Spot"); ax3.set_ylabel("Vanna nocional (sum OI)")

plt.tight_layout()
plt.show()

# Resumen útil en consola
gex_net = gex_by_strike.sum()
print(f"\nSpot SPY: {S0:.2f}")
print(f"GEX neto: {gex_net:,.0f} | Gamma flip ~ strike {gamma_flip}")
print("Gamma walls (top):", list(gamma_walls.index[:4]))
