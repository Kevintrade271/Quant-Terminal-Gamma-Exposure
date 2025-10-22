import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import yfinance as yf
import datetime as dt


def normal_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def rf_from_irx(default=0.045):
    try:
        irx_ticker = yf.Ticker("^IRX")
        hist = irx_ticker.history(period="5d")
        if not hist.empty:
            return float(hist.Close.dropna().iloc[-1]) / 100.0
        return float(default)
    except Exception:
        return float(default)


def yearfrac_to_close(expiration_str: str) -> float:
    close_dt = dt.datetime.strptime(expiration_str, "%Y-%m-%d") + dt.timedelta(hours=20)
    now = dt.datetime.utcnow()
    T = (close_dt - now).total_seconds() / (365 * 24 * 3600)
    return max(T, 1 / 365)


def dealer_sign_vector(option_type_array: np.ndarray, convention: str = 'reverse_spotgamma'):
    if convention == 'spotgamma':
        return np.where(option_type_array == 'call', -1.0, +1.0)
    if convention == 'reverse_spotgamma':
        return np.where(option_type_array == 'call', +1.0, -1.0)
    if convention == 'short_both':
        return np.full_like(option_type_array, -1.0, dtype=float)
    return np.where(option_type_array == 'call', -1.0, +1.0)


def resolve_spot_opt(ticker_in: str) -> Tuple[str, str, str]:
    t = (ticker_in or "").upper()
    presets = {
        "SPX": ("^GSPC", "SPY", "SPX (SPY opts)"),
        "^GSPC": ("^GSPC", "SPY", "SPX (SPY opts)"),
        "SPY": ("SPY", "SPY", "SPY"),
        "NDX": ("^NDX", "QQQ", "NDX (QQQ opts)"),
        "^NDX": ("^NDX", "QQQ", "NDX (QQQ opts)"),
        "QQQ": ("QQQ", "QQQ", "QQQ"),
        "GLD": ("GLD", "GLD", "GLD"),
        "XAU": ("GLD", "GLD", "Gold (GLD opts)"),
        "NAS100": ("^NDX", "QQQ", "NDX (QQQ opts)"),
        "US500": ("^GSPC", "SPY", "SPX (SPY opts)"),
    }
    spot, opt, label = presets.get(t, (t, t, t))
    if spot in ("^GSPC", "^NDX") and opt == spot:
        opt = "SPY" if spot == "^GSPC" else "QQQ"
        label = f"{spot} ({opt} opts)"
    return spot, opt, label


def fetch_spot(spot_ticker: str) -> float:
    t = yf.Ticker(spot_ticker)
    h = t.history(period="5d", interval="1m")
    if not h.empty:
        return float(h["Close"].dropna().iloc[-1])
    h = t.history(period="1d")
    if h.empty:
        raise RuntimeError(f"No price data for {spot_ticker}")
    return float(h["Close"].iloc[-1])


def fetch_option_chain(opt_ticker: str, max_weekly: int = 2, max_monthly: int = 3) -> Tuple[pd.DataFrame, list]:
    t = yf.Ticker(opt_ticker)
    expir = list(t.options) if getattr(t, "options", None) else []
    if not expir:
        raise RuntimeError(f"{opt_ticker} has no expirations available")

    def opex_list(y0: int, years: int = 2):
        out = []
        for y in range(y0, y0 + years):
            for m in range(1, 13):
                candidates = [dt.date(y, m, d) for d in range(15, 22)
                            if dt.date(y, m, d).weekday() == 4]
                if candidates:
                    out.append(candidates[0].strftime("%Y-%m-%d"))
        return out

    expir_list = list(expir)
    weekly = list(expir_list[:max_weekly])
    opex_candidates = [d for d in opex_list(dt.date.today().year) if d in expir_list]
    selected_expirs = sorted(list(set(weekly + opex_candidates[:max_monthly])))

    frames = []
    for exp_date in selected_expirs:
        try:
            chain = t.option_chain(exp_date)
            calls, puts = chain.calls.copy(), chain.puts.copy()
            if not calls.empty and not puts.empty:
                calls["option_type"], puts["option_type"] = "call", "put"
                df = pd.concat([calls, puts], ignore_index=True)
                df["expiration"] = exp_date
                frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"Could not download options for {opt_ticker}")
    return pd.concat(frames, ignore_index=True), selected_expirs


def clean_chain(chain: pd.DataFrame, spot: float,
                iv_floor: float = 0.03, iv_cap: float = 3.0,
                oi_min: int = 10, max_moneyness: float = 0.25) -> pd.DataFrame:
    df = chain.copy()

    required_cols = ["impliedVolatility", "strike", "openInterest", "option_type", "expiration"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")

    df["time_to_expiry"] = df["expiration"].apply(yearfrac_to_close).astype(float)

    iv = pd.to_numeric(df["impliedVolatility"], errors="coerce").ffill().bfill()
    df["iv"] = np.clip(iv.values.astype(float), iv_floor, iv_cap)

    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(float)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(float)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").astype(float)
    df["option_type"] = df["option_type"].astype(str)

    lo, hi = spot * (1 - max_moneyness), spot * (1 + max_moneyness)
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    df = df[(df["openInterest"] >= oi_min) & (df["time_to_expiry"] > 0)]
    df.reset_index(drop=True, inplace=True)

    return df


def gex_components_at_spot(df: pd.DataFrame, spot: float, r: float,
                          dealer_convention: str, lambda_T: float = 3.0):
    K = df["strike"].values.astype(float)
    T = df["time_to_expiry"].values.astype(float)
    iv = df["iv"].values.astype(float)
    oi = df["openInterest"].values.astype(float)
    option_type = df["option_type"].values

    with np.errstate(all="ignore"):
        d1 = (np.log(spot / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))

    gamma = normal_pdf(d1) / (spot * iv * np.sqrt(T))
    sign = dealer_sign_vector(option_type, dealer_convention)
    time_weight = np.exp(-lambda_T * T)
    gex_values = gamma * (spot ** 2) * 0.01 * 100.0 * oi * sign * time_weight

    tmp = pd.DataFrame({
        "strike": K,
        "option_type": option_type,
        "gex": gex_values
    })

    total_gex = tmp.groupby("strike")["gex"].sum()
    calls_gex = tmp[tmp["option_type"] == "call"].groupby("strike")["gex"].sum()
    puts_gex = tmp[tmp["option_type"] == "put"].groupby("strike")["gex"].sum()

    return total_gex.sort_index(), calls_gex.sort_index(), puts_gex.sort_index()


def volume_oi_by_strike(df: pd.DataFrame):
    grouped_vol = df.groupby(["strike", "option_type"])["volume"].sum().unstack(fill_value=0)
    grouped_oi = df.groupby(["strike", "option_type"])["openInterest"].sum().unstack(fill_value=0)

    calls_vol = grouped_vol.get("call", pd.Series(dtype=float)).sort_index()
    puts_vol = grouped_vol.get("put", pd.Series(dtype=float)).sort_index()
    calls_oi = grouped_oi.get("call", pd.Series(dtype=float)).sort_index()
    puts_oi = grouped_oi.get("put", pd.Series(dtype=float)).sort_index()

    return calls_vol, puts_vol, calls_oi, puts_oi


def aggregate_gamma(abs_calls: pd.Series, abs_puts: pd.Series) -> pd.Series:
    all_strikes = sorted(set(abs_calls.index).union(set(abs_puts.index)))
    ac = abs_calls.reindex(all_strikes).fillna(0.0).abs()
    ap = abs_puts.reindex(all_strikes).fillna(0.0).abs()
    return (ac + ap)


def smooth_series(s: pd.Series, span: int = 5) -> pd.Series:
    s = pd.Series(s).astype(float)
    return s.rolling(window=span, center=True, min_periods=1).mean()


def gamma_profile_curve(df: pd.DataFrame, spot: float, r: float,
                        dealer_convention: str, lambda_T: float,
                        span_pct: float = 0.12, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    lo, hi = spot * (1 - span_pct), spot * (1 + span_pct)
    price_grid = np.linspace(lo, hi, n_points)

    K = df["strike"].values.astype(float)
    T = df["time_to_expiry"].values.astype(float)
    iv = df["iv"].values.astype(float)
    oi = df["openInterest"].values.astype(float)
    option_type = df["option_type"].values

    def gex_total_at_price(S: float) -> float:
        with np.errstate(all="ignore"):
            d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        gamma = normal_pdf(d1) / (S * iv * np.sqrt(T))
        sign = dealer_sign_vector(option_type, dealer_convention)
        time_weight = np.exp(-lambda_T * T)
        return float(np.nansum(gamma * (S ** 2) * 0.01 * 100.0 * oi * sign * time_weight))

    profile = np.array([gex_total_at_price(S) for S in price_grid])

    sign_changes = np.sign(profile)
    idx = np.where(np.diff(sign_changes) != 0)[0]
    flip = None

    if len(idx):
        i = idx[np.argmin(abs(price_grid[idx] - spot))]
        flip = float(price_grid[i] - profile[i] * (price_grid[i+1] - price_grid[i]) / (profile[i+1] - profile[i]))

    return price_grid, profile, flip


def build_enhanced_gex_data(ticker: str, odte_only: bool = False,
                           max_weekly: int = 2, max_monthly: int = 3,
                           dealer_convention: str = 'reverse_spotgamma',
                           lambda_T: float = 3.0) -> Dict:
    spot_ticker, opt_ticker, label = resolve_spot_opt(ticker)
    spot = fetch_spot(spot_ticker)
    r = rf_from_irx()

    chain_raw, _ = fetch_option_chain(opt_ticker, max_weekly, max_monthly)
    df_all = clean_chain(chain_raw, spot)

    if odte_only:
        today = dt.datetime.utcnow().date()
        df_all = df_all.assign(
            DTE=df_all['expiration'].apply(
                lambda v: (dt.datetime.strptime(v, "%Y-%m-%d").date() - today).days
            )
        )
        dte0 = df_all.loc[df_all['DTE'] >= 0, 'DTE'].min()
        if pd.isna(dte0):
            dte0 = df_all['DTE'].iloc[(df_all['DTE'].abs()).argsort().values[0]]
        df_all = df_all[df_all['DTE'] == dte0].copy()

    total_gex, calls_gex, puts_gex = gex_components_at_spot(df_all, spot, r, dealer_convention, lambda_T)
    calls_vol, puts_vol, calls_oi, puts_oi = volume_oi_by_strike(df_all)
    price_grid, gamma_profile, gamma_flip = gamma_profile_curve(df_all, spot, r, dealer_convention, lambda_T)

    lo, hi = spot * 0.95, spot * 1.05
    all_strikes = sorted(set(total_gex.index) | set(calls_vol.index) | set(puts_vol.index))
    strikes = np.array(all_strikes, dtype=float)
    mask = (strikes >= lo) & (strikes <= hi)

    if not mask.any():
        mask = np.ones_like(strikes, dtype=bool)

    strikes_windowed = strikes[mask]

    total_gex_windowed = total_gex.reindex(strikes_windowed).fillna(0.0)
    calls_gex_windowed = calls_gex.reindex(strikes_windowed).fillna(0.0)
    puts_gex_windowed = puts_gex.reindex(strikes_windowed).fillna(0.0)

    ag_series = smooth_series(
        aggregate_gamma(calls_gex_windowed, puts_gex_windowed),
        span=5
    )
    cv_series = smooth_series(calls_vol.reindex(strikes_windowed).fillna(0.0), span=5)
    pv_series = smooth_series(puts_vol.reindex(strikes_windowed).fillna(0.0), span=5)

    return {
        "spot": float(spot),
        "ticker": ticker.upper(),
        "label": label,
        "strikes": strikes_windowed.tolist(),
        "net_gex": total_gex_windowed.tolist(),
        "net_gex_positive": total_gex_windowed.where(total_gex_windowed > 0, 0.0).tolist(),
        "net_gex_negative": total_gex_windowed.where(total_gex_windowed < 0, 0.0).tolist(),
        "absolute_gamma": ag_series.tolist(),
        "call_volume": cv_series.tolist(),
        "put_volume": pv_series.tolist(),
        "calls_oi": calls_oi.reindex(strikes_windowed).fillna(0.0).tolist(),
        "puts_oi": puts_oi.reindex(strikes_windowed).fillna(0.0).tolist(),
        "gamma_flip": float(gamma_flip) if gamma_flip is not None else None,
        "gamma_profile": {
            "price_grid": price_grid.tolist(),
            "profile": gamma_profile.tolist()
        },
        "window_range": {"low": float(lo), "high": float(hi)}
    }
