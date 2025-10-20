export interface GreekDataPoint {
  exp: string;
  K: number;
  side: 'C' | 'P';
  GEX: number;
  CHARM: number;
}

export interface GreeksResponse {
  spot: number;
  data: GreekDataPoint[];
  ticker: string;
  timestamp: string;
  gamma_flip?: number;
  call_wall?: number;
  put_wall?: number;
}

export interface VolatilityResponse {
  spot: number;
  matrix: Record<string, Record<string, number | null>>;
  vix_current: number;
  vix_zscore: number;
  ticker: string;
  timestamp: string;
}

export interface StatusResponse {
  spot: number;
  vix_current: number;
  vix_zscore: number;
  ticker: string;
  timestamp: string;
}

export interface GammaLevels {
  gamma_flip: number | null;
  call_wall: number;
  put_wall: number;
  call_wall_gex: number;
  put_wall_gex: number;
}

export interface IVSkewDataPoint {
  exp: string;
  strike: number;
  moneyness: number;
  iv: number;
  side: string;
  oi: number;
}

export interface IVSkewResponse {
  spot: number;
  data: IVSkewDataPoint[];
  ticker: string;
  timestamp: string;
}

export interface NetGammaProfile {
  strikes: number[];
  net_gamma: number[];
}

export interface OperationalLevels {
  slip_risk: number | null;
  gamma_pin: number | null;
  exhaustion: number | null;
  convex_hotspot: number | null;
}

export interface VolumeByStrike {
  [strike: string]: {
    call_volume: number;
    put_volume: number;
    call_oi: number;
    put_oi: number;
  };
}

export interface AdvancedAnalyticsResponse {
  spot: number;
  ticker: string;
  timestamp: string;
  gamma_levels: {
    gamma_flip: number | null;
    call_wall: number;
    put_wall: number;
    call_wall_gex: number;
    put_wall_gex: number;
  };
  net_gamma_profile: NetGammaProfile;
  operational_levels: OperationalLevels;
  volume_by_strike: VolumeByStrike;
}

export type TabValue = 'gex-charm' | 'volatility' | 'iv-skew' | 'advanced';

export const AVAILABLE_TICKERS = [
  { value: 'SPY', label: 'SPY - S&P 500 ETF' },
  { value: 'QQQ', label: 'QQQ - Nasdaq 100 ETF' },
  { value: 'DIA', label: 'DIA - Dow Jones ETF' },
  { value: 'IWM', label: 'IWM - Russell 2000 ETF' },
  { value: 'GLD', label: 'GLD - Gold ETF' },
  { value: 'SLV', label: 'SLV - Silver ETF' },
] as const;
