from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
import pandas as pd
from ..services.volatility_calculator import load_and_build_matrix, get_vix_data, z_last
from ..models.schemas import VolatilityResponse, ErrorResponse

router = APIRouter()


@router.get("/volatility/{ticker}", response_model=VolatilityResponse)
async def get_volatility(
    ticker: str,
    max_exp: int = Query(default=15, ge=1, le=30),
    strike_span: float = Query(default=0.10, ge=0.05, le=0.30),
    max_cols: int = Query(default=40, ge=10, le=100),
    min_oi: int = Query(default=100, ge=0)
):
    try:
        pivot_df, spot = load_and_build_matrix(
            ticker=ticker.upper(),
            max_exp=max_exp,
            strike_span=strike_span,
            max_cols=max_cols,
            min_oi=min_oi
        )

        vix_hist, current_vix = get_vix_data()
        vix_zscore = z_last(vix_hist.iloc[:-1].values, current_vix)

        matrix_dict = {}
        for idx in pivot_df.index:
            matrix_dict[str(idx)] = {
                str(col): float(val) if not pd.isna(val) else None
                for col, val in pivot_df.loc[idx].items()
            }

        return VolatilityResponse(
            spot=spot,
            matrix=matrix_dict,
            vix_current=current_vix,
            vix_zscore=vix_zscore,
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
