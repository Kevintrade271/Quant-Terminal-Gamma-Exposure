import axios from 'axios';
import type {
  GreeksResponse,
  VolatilityResponse,
  StatusResponse,
  IVSkewResponse,
  AdvancedAnalyticsResponse,
  GexEnhancedResponse
} from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function fetchGreeks(ticker: string = 'SPY', odteOnly: boolean = false): Promise<GreeksResponse> {
  const response = await api.get(`/api/greeks/${ticker}`, {
    params: { odte_only: odteOnly }
  });
  return response.data;
}

export async function fetchVolatility(ticker: string = 'SPY'): Promise<VolatilityResponse> {
  const response = await api.get(`/api/volatility/${ticker}`);
  return response.data;
}

export async function fetchStatus(ticker: string = 'SPY'): Promise<StatusResponse> {
  const response = await api.get(`/api/status/${ticker}`);
  return response.data;
}

export async function fetchIVSkew(ticker: string = 'SPY'): Promise<IVSkewResponse> {
  const response = await api.get(`/api/iv-skew/${ticker}`);
  return response.data;
}

export async function fetchAdvancedAnalytics(ticker: string = 'SPY', odteOnly: boolean = false): Promise<AdvancedAnalyticsResponse> {
  const response = await api.get(`/api/advanced-analytics/${ticker}`, {
    params: { odte_only: odteOnly }
  });
  return response.data;
}

export async function fetchGexEnhanced(ticker: string = 'SPY', odteOnly: boolean = false): Promise<GexEnhancedResponse> {
  const response = await api.get(`/api/gex-enhanced/${ticker}`, {
    params: { odte_only: odteOnly }
  });
  return response.data;
}
