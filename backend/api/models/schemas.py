from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class GreekDataPoint(BaseModel):
    exp: date
    K: float
    side: str
    GEX: float
    CHARM: float


class GreeksResponse(BaseModel):
    spot: float
    data: List[GreekDataPoint]
    ticker: str
    timestamp: str


class VolatilityDataPoint(BaseModel):
    exp: date
    strike: float
    iv: float


class VolatilityResponse(BaseModel):
    spot: float
    matrix: dict
    vix_current: float
    vix_zscore: float
    ticker: str
    timestamp: str


class StatusResponse(BaseModel):
    spot: float
    vix_current: float
    vix_zscore: float
    ticker: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
