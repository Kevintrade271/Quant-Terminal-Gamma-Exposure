import pandas as pd
import yfinance as yf
from typing import Tuple, List, Dict


def calculate_iv_skew(
    ticker: str = "SPY",
    max_exp: int = 3
) -> Tuple[float, List[Dict]]:
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

    for exp in exps:
        try:
            oc = tk.option_chain(exp)
        except Exception:
            continue

        for side, df in (("calls", oc.calls), ("puts", oc.puts)):
            if df is None or df.empty:
                continue

            df = df[["strike", "impliedVolatility", "openInterest"]].copy()
            df = df.dropna()
            df = df[df.openInterest >= 50]
            df = df[df.impliedVolatility > 0]
            df = df[(df.strike >= spot * 0.85) & (df.strike <= spot * 1.15)]

            if df.empty:
                continue

            for _, row in df.iterrows():
                moneyness = float(row.strike) / spot
                rows.append({
                    "exp": pd.to_datetime(exp).date(),
                    "strike": float(row.strike),
                    "moneyness": moneyness,
                    "iv": float(row.impliedVolatility),
                    "side": side,
                    "oi": float(row.openInterest)
                })

    return spot, rows
