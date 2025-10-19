from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from ..services.greeks_calculator import build_greeks_df
from ..services.volatility_calculator import get_vix_data, z_last
from ..services.iv_skew_calculator import calculate_iv_skew
from ..services.advanced_analytics import get_advanced_analytics
from ..models.schemas import (
    GreeksResponse, GreekDataPoint, StatusResponse, ErrorResponse,
    IVSkewResponse, IVSkewDataPoint, AdvancedAnalyticsResponse
)

router = APIRouter()


@router.get("/greeks/{ticker}", response_model=GreeksResponse)
async def get_greeks(
    ticker: str,
    max_exp: int = Query(default=8, ge=1, le=20),
    r: float = Query(default=0.05, ge=0, le=0.2),
    q: float = Query(default=0.015, ge=0, le=0.1),
    min_oi: int = Query(default=100, ge=0),
    odte_only: bool = Query(default=False)
):
    try:
        spot, df, gamma_levels = build_greeks_df(
            ticker=ticker.upper(),
            max_exp=max_exp,
            r=r,
            q=q,
            min_oi=min_oi,
            odte_only=odte_only
        )

        data_points = [
            GreekDataPoint(**row)
            for row in df.to_dict('records')
        ]

        return GreeksResponse(
            spot=spot,
            data=data_points,
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat(),
            gamma_flip=gamma_levels.get("gamma_flip"),
            call_wall=gamma_levels.get("call_wall"),
            put_wall=gamma_levels.get("put_wall"),
            call_wall_gex=gamma_levels.get("call_wall_gex"),
            put_wall_gex=gamma_levels.get("put_wall_gex")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{ticker}", response_model=StatusResponse)
async def get_status(ticker: str):
    try:
        spot, _, _ = build_greeks_df(
            ticker=ticker.upper(),
            max_exp=2,
            r=0.05,
            q=0.015,
            min_oi=200
        )

        vix_hist, current_vix = get_vix_data()
        vix_zscore = z_last(vix_hist.iloc[:-1].values, current_vix)

        return StatusResponse(
            spot=spot,
            vix_current=current_vix,
            vix_zscore=vix_zscore,
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/iv-skew/{ticker}", response_model=IVSkewResponse)
async def get_iv_skew(
    ticker: str,
    max_exp: int = Query(default=3, ge=1, le=10)
):
    try:
        spot, data = calculate_iv_skew(
            ticker=ticker.upper(),
            max_exp=max_exp
        )

        data_points = [IVSkewDataPoint(**row) for row in data]

        return IVSkewResponse(
            spot=spot,
            data=data_points,
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/{ticker}", response_model=AdvancedAnalyticsResponse)
async def get_advanced_analytics_endpoint(
    ticker: str,
    max_exp: int = Query(default=6, ge=1, le=20),
    r: float = Query(default=0.045, ge=0, le=0.2),
    q: float = Query(default=0.012, ge=0, le=0.1),
    min_oi: int = Query(default=200, ge=0),
    odte_only: bool = Query(default=False)
):
    try:
        analytics = get_advanced_analytics(
            ticker=ticker.upper(),
            max_exp=max_exp,
            r=r,
            q=q,
            min_oi=min_oi,
            odte_only=odte_only
        )

        return AdvancedAnalyticsResponse(
            spot=analytics["spot"],
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat(),
            gamma_levels=analytics["gamma_levels"],
            net_gamma_profile=analytics["net_gamma_profile"],
            operational_levels=analytics["operational_levels"],
            volume_by_strike=analytics["volume_by_strike"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
