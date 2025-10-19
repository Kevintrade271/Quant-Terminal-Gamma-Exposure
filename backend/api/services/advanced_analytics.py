import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Dict, List
from scipy.interpolate import interp1d
from .greeks_calculator import build_greeks_df


def calculate_net_gamma_profile(
    spot: float,
    df: pd.DataFrame,
    num_points: int = 100
) -> Dict:
    """
    Calculate continuous net gamma profile
    """
    if df.empty:
        return {"strikes": [], "net_gamma": []}

    gex_by_strike = df.groupby('K')['GEX'].sum().sort_index()

    if len(gex_by_strike) < 2:
        return {"strikes": [], "net_gamma": []}

    strikes = gex_by_strike.index.values
    gex_values = gex_by_strike.values

    strike_range = np.linspace(strikes.min(), strikes.max(), num_points)

    try:
        f = interp1d(strikes, gex_values, kind='cubic', fill_value='extrapolate')
        smooth_gex = f(strike_range)
    except Exception:
        f = interp1d(strikes, gex_values, kind='linear', fill_value='extrapolate')
        smooth_gex = f(strike_range)

    return {
        "strikes": strike_range.tolist(),
        "net_gamma": smooth_gex.tolist()
    }


def calculate_volume_by_strike(
    ticker: str,
    max_exp: int = 3
) -> Dict[float, Dict]:
    """
    Calculate volume and open interest by strike
    """
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

    volume_data = {}

    for exp in exps:
        try:
            oc = tk.option_chain(exp)
        except Exception:
            continue

        for side, df_opt in (("calls", oc.calls), ("puts", oc.puts)):
            if df_opt is None or df_opt.empty:
                continue

            df_opt = df_opt[["strike", "volume", "openInterest"]].copy()
            df_opt = df_opt.dropna()

            for _, row in df_opt.iterrows():
                strike = float(row.strike)

                if strike not in volume_data:
                    volume_data[strike] = {
                        "call_volume": 0,
                        "put_volume": 0,
                        "call_oi": 0,
                        "put_oi": 0
                    }

                if side == "calls":
                    volume_data[strike]["call_volume"] += float(row.get("volume", 0))
                    volume_data[strike]["call_oi"] += float(row.get("openInterest", 0))
                else:
                    volume_data[strike]["put_volume"] += float(row.get("volume", 0))
                    volume_data[strike]["put_oi"] += float(row.get("openInterest", 0))

    return volume_data


def identify_operational_levels(
    df: pd.DataFrame,
    spot: float
) -> Dict:
    """
    Identify key operational levels:
    - Slip Risk (put wall)
    - Gamma Pin (max gamma concentration near spot)
    - Exhaustion (call wall)
    - Convex Hotspot (max curvature above spot)
    """
    if df.empty:
        return {
            "slip_risk": None,
            "gamma_pin": None,
            "exhaustion": None,
            "convex_hotspot": None
        }

    gex_by_strike = df.groupby('K')['GEX'].sum().sort_index()

    slip_risk = float(gex_by_strike.idxmax())
    exhaustion = float(gex_by_strike.idxmin())

    window = gex_by_strike[(gex_by_strike.index >= spot * 0.98) &
                           (gex_by_strike.index <= spot * 1.02)]
    gamma_pin = float(window.abs().idxmax()) if not window.empty else spot

    gex_abs = gex_by_strike.abs().sort_index()
    curvature = (gex_abs.shift(-1) - 2 * gex_abs + gex_abs.shift(1)).abs()
    curvature_above = curvature[curvature.index > spot]
    convex_hotspot = float(curvature_above.idxmax()) if not curvature_above.empty else None

    return {
        "slip_risk": slip_risk,
        "gamma_pin": gamma_pin,
        "exhaustion": exhaustion,
        "convex_hotspot": convex_hotspot
    }


def calculate_gamma_flip_point(gex_by_strike: pd.Series) -> float:
    """
    Find the gamma flip point where net GEX crosses zero
    """
    for i in range(len(gex_by_strike) - 1):
        if (gex_by_strike.iloc[i] > 0 and gex_by_strike.iloc[i + 1] < 0) or \
           (gex_by_strike.iloc[i] < 0 and gex_by_strike.iloc[i + 1] > 0):
            return float(gex_by_strike.index[i])

    return None


def get_advanced_analytics(
    ticker: str = "SPY",
    max_exp: int = 6,
    r: float = 0.045,
    q: float = 0.012,
    min_oi: int = 200,
    odte_only: bool = False
) -> Dict:
    """
    Get all advanced analytics in one call
    """
    spot, df, gamma_levels = build_greeks_df(
        ticker=ticker,
        max_exp=max_exp,
        r=r,
        q=q,
        min_oi=min_oi,
        odte_only=odte_only
    )

    net_gamma_profile = calculate_net_gamma_profile(spot, df)

    operational_levels = identify_operational_levels(df, spot)

    try:
        volume_data = calculate_volume_by_strike(ticker, max_exp=3)
    except Exception:
        volume_data = {}

    return {
        "spot": spot,
        "gamma_levels": gamma_levels,
        "net_gamma_profile": net_gamma_profile,
        "operational_levels": operational_levels,
        "volume_by_strike": volume_data
    }
