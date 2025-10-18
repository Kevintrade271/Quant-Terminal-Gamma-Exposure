from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from ..services.greeks_calculator import build_greeks_df
from ..services.volatility_calculator import get_vix_data, z_last
from ..models.schemas import GreeksResponse, GreekDataPoint, StatusResponse, ErrorResponse

router = APIRouter()


@router.get("/greeks/{ticker}", response_model=GreeksResponse)
async def get_greeks(
    ticker: str,
    max_exp: int = Query(default=8, ge=1, le=20),
    r: float = Query(default=0.05, ge=0, le=0.2),
    q: float = Query(default=0.015, ge=0, le=0.1),
    min_oi: int = Query(default=100, ge=0)
):
    try:
        spot, df = build_greeks_df(
            ticker=ticker.upper(),
            max_exp=max_exp,
            r=r,
            q=q,
            min_oi=min_oi
        )

        data_points = [
            GreekDataPoint(**row)
            for row in df.to_dict('records')
        ]

        return GreeksResponse(
            spot=spot,
            data=data_points,
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{ticker}", response_model=StatusResponse)
async def get_status(ticker: str):
    try:
        spot, _ = build_greeks_df(
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
