import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Optional


def z_last(hist_vals, now):
    x = np.array(list(hist_vals) + [now], dtype=float)
    if x.std() < 1e-9:
        return 0.0
    return float((x - x.mean())[-1] / x.std())


def get_vix_data() -> Tuple[pd.Series, float]:
    vix_tk = yf.Ticker("^VIX")
    vix_hist = vix_tk.history(period="6mo")
    if vix_hist.empty:
        raise RuntimeError("No se pudo obtener el histórico del VIX.")
    current_vix = float(vix_hist["Close"].iloc[-1])
    return vix_hist["Close"], current_vix


def safe_mean(a, b):
    vals = [x for x in (a, b) if (pd.notna(x) and x > 0)]
    return float(np.mean(vals)) if vals else np.nan


def nearest(arr, value):
    a = np.asarray(arr, dtype=float)
    return a[(np.abs(a - value)).argmin()]


def load_and_build_matrix(
    ticker: str = "SPY",
    max_exp: int = 15,
    strike_span: float = 0.10,
    max_cols: int = 40,
    min_oi: int = 100
) -> Tuple[pd.DataFrame, float]:
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
                if c not in df.columns:
                    df[c] = np.nan
            df = df.dropna(subset=["impliedVolatility", "strike", "openInterest"])
            if min_oi > 0:
                df = df[df["openInterest"] >= min_oi]
            df = df[(df["strike"] >= lo_k) & (df["strike"] <= hi_k)]
            if side_name == "calls":
                calls = df
            else:
                puts = df

        if calls.empty and puts.empty:
            continue

        c = calls[["strike", "impliedVolatility"]].rename(columns={"impliedVolatility": "iv_c"})
        p = puts[["strike", "impliedVolatility"]].rename(columns={"impliedVolatility": "iv_p"})
        m = pd.merge(c, p, on="strike", how="outer")
        m["iv"] = m.apply(lambda r: safe_mean(r["iv_c"], r["iv_p"]), axis=1)
        m = m.dropna(subset=["iv"])
        if m.empty:
            continue
        m["exp"] = pd.to_datetime(e).date()
        rows.append(m[["exp", "strike", "iv"]])

    if not rows:
        return pd.DataFrame(), spot

    big = pd.concat(rows, ignore_index=True)
    strikes = np.sort(big["strike"].unique())
    if len(strikes) > max_cols:
        atm_k = nearest(strikes, spot)
        order = np.argsort(np.abs(strikes - atm_k))[:max_cols]
        keep_strikes = np.sort(strikes[order])
        big = big[big["strike"].isin(keep_strikes)]

    pivot = big.pivot_table(
        index="exp",
        columns="strike",
        values="iv",
        aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    return pivot, spot
