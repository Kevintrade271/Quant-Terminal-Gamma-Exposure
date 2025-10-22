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
    gamma_flip: Optional[float] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    call_wall_gex: Optional[float] = None
    put_wall_gex: Optional[float] = None


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


class IVSkewDataPoint(BaseModel):
    exp: date
    strike: float
    moneyness: float
    iv: float
    side: str
    oi: float


class IVSkewResponse(BaseModel):
    spot: float
    data: List[IVSkewDataPoint]
    ticker: str
    timestamp: str


class NetGammaProfile(BaseModel):
    strikes: List[float]
    net_gamma: List[float]


class OperationalLevels(BaseModel):
    slip_risk: Optional[float] = None
    gamma_pin: Optional[float] = None
    exhaustion: Optional[float] = None
    convex_hotspot: Optional[float] = None


class VolumeData(BaseModel):
    strike: float
    call_volume: float
    put_volume: float
    call_oi: float
    put_oi: float


class AdvancedAnalyticsResponse(BaseModel):
    spot: float
    ticker: str
    timestamp: str
    gamma_levels: dict
    net_gamma_profile: dict
    operational_levels: dict
    volume_by_strike: dict


class GammaProfileData(BaseModel):
    price_grid: List[float]
    profile: List[float]


class WindowRange(BaseModel):
    low: float
    high: float


class GexEnhancedResponse(BaseModel):
    spot: float
    ticker: str
    label: str
    strikes: List[float]
    net_gex: List[float]
    net_gex_positive: List[float]
    net_gex_negative: List[float]
    absolute_gamma: List[float]
    call_volume: List[float]
    put_volume: List[float]
    calls_oi: List[float]
    puts_oi: List[float]
    gamma_flip: Optional[float] = None
    gamma_profile: GammaProfileData
    window_range: WindowRange
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
