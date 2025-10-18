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


def build_greeks_df(
    ticker: str = "SPY",
    max_exp: int = 6,
    r: float = 0.045,
    q: float = 0.012,
    min_oi: int = 200
) -> Tuple[float, pd.DataFrame]:
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
    rows = []

    for e in exps:
        exp_ts = pd.to_datetime(e)
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

    return spot, pd.DataFrame(rows)
