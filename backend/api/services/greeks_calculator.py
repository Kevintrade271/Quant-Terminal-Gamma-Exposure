import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
from typing import Tuple, Optional


def bs_d1(S: float, K: float, T: float, r: float, q: float, iv: float) -> float:
    return (np.log(S / K) + (r - q + 0.5 * iv * iv) * T) / (iv * np.sqrt(T) + 1e-12)


def bs_gamma(S: float, K: float, T: float, r: float, q: float, iv: float) -> float:
    d1 = bs_d1(S, K, T, r, q, iv)
    return np.exp(-q * T) * norm.pdf(d1) / (S * iv * np.sqrt(T) + 1e-12)


def bs_charm(S: float, K: float, T: float, r: float, q: float, iv: float, cp: str) -> float:
    d1 = bs_d1(S, K, T, r, q, iv)
    d2 = d1 - iv * np.sqrt(T)
    s = 1 if cp == "C" else -1
    t1 = -q * np.exp(-q * T) * norm.cdf(s * d1) * s
    t2 = np.exp(-q * T) * norm.pdf(d1) * (2 * (r - q) * T - d2 * iv * np.sqrt(T)) / (2 * T * iv * np.sqrt(T) + 1e-12)
    return t1 + t2


def _load_chain(ticker: str, exp: str):
    tk = yf.Ticker(ticker)
    oc = tk.option_chain(exp)
    for df in (oc.calls, oc.puts):
        for c in ["impliedVolatility", "strike", "openInterest", "bid", "ask"]:
            if c not in df.columns:
                df[c] = np.nan
    return oc


def calculate_gamma_levels(df: pd.DataFrame, spot: float) -> dict:
    """Calculate key gamma levels: flip point, call wall, put wall"""
    if df.empty:
        return {
            "gamma_flip": None,
            "call_wall": spot,
            "put_wall": spot,
            "call_wall_gex": 0,
            "put_wall_gex": 0
        }

    # Aggregate GEX by strike
    gex_by_strike = df.groupby('K')['GEX'].sum().sort_index()

    # Find gamma flip point (where net GEX crosses zero)
    gamma_flip = None
    for i in range(len(gex_by_strike) - 1):
        if (gex_by_strike.iloc[i] > 0 and gex_by_strike.iloc[i + 1] < 0) or \
           (gex_by_strike.iloc[i] < 0 and gex_by_strike.iloc[i + 1] > 0):
            gamma_flip = float(gex_by_strike.index[i])
            break

    # Find call wall (highest positive GEX above spot)
    calls_above_spot = gex_by_strike[gex_by_strike.index >= spot]
    if not calls_above_spot.empty and calls_above_spot.max() > 0:
        call_wall_idx = calls_above_spot.idxmax()
        call_wall = float(call_wall_idx)
        call_wall_gex = float(calls_above_spot.max())
    else:
        call_wall = spot * 1.05
        call_wall_gex = 0

    # Find put wall (highest negative GEX below spot)
    puts_below_spot = gex_by_strike[gex_by_strike.index <= spot]
    if not puts_below_spot.empty and puts_below_spot.min() < 0:
        put_wall_idx = puts_below_spot.idxmin()
        put_wall = float(put_wall_idx)
        put_wall_gex = float(abs(puts_below_spot.min()))
    else:
        put_wall = spot * 0.95
        put_wall_gex = 0

    return {
        "gamma_flip": gamma_flip,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_wall_gex": call_wall_gex,
        "put_wall_gex": put_wall_gex
    }


def build_greeks_df(
    ticker: str = "SPY",
    max_exp: int = 6,
    r: float = 0.045,
    q: float = 0.012,
    min_oi: int = 200,
    odte_only: bool = False
) -> Tuple[float, pd.DataFrame, dict]:
    tk = yf.Ticker(ticker)
    try:
        spot = float(tk.fast_info["last_price"])
    except Exception:
        hist = tk.history(period="1d")
        if hist.empty:
            raise RuntimeError("No se pudo obtener el spot.")
        spot = float(hist["Close"].iloc[-1])

    exps = tk.options[:max_exp] if tk.options else []
    if not exps:
        raise RuntimeError("Sin expiraciones disponibles.")

    now = pd.Timestamp.utcnow().tz_localize(None)
    today = now.date()
    rows = []

    for e in exps:
        exp_ts = pd.to_datetime(e)
        exp_date = exp_ts.date()

        # Filter for ODTE if requested
        if odte_only and exp_date != today:
            continue

        T = max((exp_ts - now).total_seconds(), 0) / (365.0 * 24 * 3600.0)
        if T <= 0:
            continue

        try:
            oc = _load_chain(ticker, e)
        except Exception:
            continue

        for side, df in (("C", oc.calls), ("P", oc.puts)):
            if df is None or df.empty:
                continue
            df = df[["strike", "impliedVolatility", "openInterest"]].dropna()
            if min_oi > 0:
                df = df[df.openInterest >= min_oi]
            df = df[(df.impliedVolatility > 0) & (df.strike > 0)]
            if df.empty:
                continue

            for _, r0 in df.iterrows():
                K = float(r0.strike)
                iv = float(r0.impliedVolatility)
                oi = float(r0.openInterest)
                gamma = bs_gamma(spot, K, T, r, q, iv)
                charm = bs_charm(spot, K, T, r, q, iv, side)
                mult = 100.0
                gex = -oi * gamma * (spot * mult)
                chm = -oi * charm * mult
                rows.append({
                    "exp": pd.to_datetime(e).date(),
                    "K": K,
                    "side": side,
                    "GEX": gex,
                    "CHARM": chm
                })

    df_result = pd.DataFrame(rows)
    gamma_levels = calculate_gamma_levels(df_result, spot)

    return spot, df_result, gamma_levels
