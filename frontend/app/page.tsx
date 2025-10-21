'use client';

import { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Tabs,
  Tab,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  ThemeProvider,
  Switch,
  FormControlLabel,
  IconButton,
  Tooltip,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { theme } from '@/lib/theme';
import { fetchGreeks, fetchVolatility, fetchStatus, fetchIVSkew, fetchAdvancedAnalytics } from '@/lib/api';
import type { GreeksResponse, VolatilityResponse, StatusResponse, IVSkewResponse, AdvancedAnalyticsResponse, TabValue } from '@/lib/types';
import { AVAILABLE_TICKERS } from '@/lib/types';
import GexCharmChart from '@/components/GexCharmChart';
import VolatilityHeatmap from '@/components/VolatilityHeatmap';
import IVSkewChart from '@/components/IVSkewChart';
import AdvancedAnalyticsView from '@/components/AdvancedAnalyticsView';
import TickerCard from '@/components/TickerCard';

const REFRESH_INTERVAL = 5 * 60 * 1000;

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabValue>('gex-charm');
  const [ticker, setTicker] = useState('SPY');
  const [odteMode, setOdteMode] = useState(false);
  const [greeksData, setGreeksData] = useState<GreeksResponse | null>(null);
  const [volatilityData, setVolatilityData] = useState<VolatilityResponse | null>(null);
  const [ivSkewData, setIvSkewData] = useState<IVSkewResponse | null>(null);
  const [advancedData, setAdvancedData] = useState<AdvancedAnalyticsResponse | null>(null);
  const [statusData, setStatusData] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const loadData = async () => {
    try {
      setError(null);

      setGreeksData(null);
      setVolatilityData(null);
      setIvSkewData(null);
      setAdvancedData(null);

      if (activeTab === 'gex-charm') {
        const [greeks, status] = await Promise.all([
          fetchGreeks(ticker, odteMode),
          fetchStatus(ticker),
        ]);
        setGreeksData(greeks);
        setStatusData(status);
      } else if (activeTab === 'volatility') {
        const [volatility, status] = await Promise.all([
          fetchVolatility(ticker),
          fetchStatus(ticker),
        ]);
        setVolatilityData(volatility);
        setStatusData(status);
      } else if (activeTab === 'iv-skew') {
        const [ivSkew, status] = await Promise.all([
          fetchIVSkew(ticker),
          fetchStatus(ticker),
        ]);
        setIvSkewData(ivSkew);
        setStatusData(status);
      } else if (activeTab === 'advanced') {
        const [advanced, status] = await Promise.all([
          fetchAdvancedAnalytics(ticker, odteMode),
          fetchStatus(ticker),
        ]);
        setAdvancedData(advanced);
        setStatusData(status);
      }

      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al cargar datos');
      console.error('Error loading data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [activeTab, ticker, odteMode]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: TabValue) => {
    setActiveTab(newValue);
    setLoading(true);
  };

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          minHeight: '100vh',
          bgcolor: 'background.default',
          py: 4,
        }}
      >
        <Container maxWidth="xl">
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', md: 'row' },
              justifyContent: 'space-between',
              alignItems: { xs: 'flex-start', md: 'center' },
              gap: 2,
              mb: 3,
            }}
          >
            <Typography
              variant="h3"
              component="h1"
              sx={{
                color: 'text.primary',
                fontWeight: 700,
                fontSize: { xs: '1.75rem', md: '3rem' },
              }}
            >
              🎯 GammaVision
            </Typography>

            {statusData && (
              <Card sx={{ bgcolor: 'background.paper', px: 3, py: 1.5, width: { xs: '100%', md: 'auto' } }}>
                <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
                  ✅ Última Actualización: {lastUpdate ? lastUpdate.toLocaleTimeString('es-ES') : 'Cargando...'}
                </Typography>
                <Typography variant="body2" color="text.primary" sx={{ fontWeight: 600, mt: 0.5 }}>
                  Spot: ${statusData.spot.toFixed(2)} | VIX: {statusData.vix_current.toFixed(2)} (Z:{' '}
                  {statusData.vix_zscore >= 0 ? '+' : ''}
                  {statusData.vix_zscore.toFixed(2)})
                </Typography>
              </Card>
            )}
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography
              variant="h6"
              sx={{
                textAlign: 'center',
                color: 'text.secondary',
                mb: 2,
                fontWeight: 600,
              }}
            >
              Selecciona un Ticker
            </Typography>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: {
                  xs: 'repeat(2, 1fr)',
                  sm: 'repeat(3, 1fr)',
                  md: 'repeat(6, 1fr)',
                },
                gap: 2,
                mb: 3,
              }}
            >
              {AVAILABLE_TICKERS.map((t) => (
                <TickerCard
                  key={t.value}
                  ticker={t.value}
                  label={t.label}
                  isSelected={ticker === t.value}
                  onClick={() => {
                    setTicker(t.value);
                    setLoading(true);
                  }}
                />
              ))}
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 3 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={odteMode}
                    onChange={(e) => {
                      setOdteMode(e.target.checked);
                      setLoading(true);
                    }}
                  />
                }
                label="Modo ODTE"
              />

              <Tooltip title="Actualizar datos manualmente">
                <IconButton
                  onClick={() => {
                    setLoading(true);
                    loadData();
                  }}
                  disabled={loading}
                  sx={{
                    color: 'primary.main',
                    '&:hover': {
                      backgroundColor: 'rgba(96, 165, 250, 0.1)',
                    },
                  }}
                >
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              variant="fullWidth"
              sx={{
                '& .MuiTab-root': {
                  color: 'text.secondary',
                  '&.Mui-selected': {
                    color: 'primary.main',
                  },
                },
              }}
            >
              <Tab label="📊 GEX & CHARM" value="gex-charm" />
              <Tab label="🌡️ Volatilidad IV" value="volatility" />
              <Tab label="📈 IV Skew" value="iv-skew" />
              <Tab label="🎯 Advanced Analytics" value="advanced" />
            </Tabs>
          </Box>

          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
              <CircularProgress size={60} />
            </Box>
          ) : (
            <Box>
              {activeTab === 'gex-charm' && greeksData && (
                <Box>
                  <Box sx={{ mb: 3 }}>
                    <GexCharmChart
                      data={greeksData.data}
                      spot={greeksData.spot}
                      ticker={greeksData.ticker}
                      valueType="GEX"
                      gammaFlip={greeksData.gamma_flip}
                      callWall={greeksData.call_wall}
                      putWall={greeksData.put_wall}
                    />
                  </Box>
                  <Box>
                    <GexCharmChart
                      data={greeksData.data}
                      spot={greeksData.spot}
                      ticker={greeksData.ticker}
                      valueType="CHARM"
                    />
                  </Box>
                </Box>
              )}

              {activeTab === 'volatility' && volatilityData && (
                <VolatilityHeatmap data={volatilityData} />
              )}

              {activeTab === 'iv-skew' && ivSkewData && (
                <IVSkewChart data={ivSkewData} />
              )}

              {activeTab === 'advanced' && advancedData && (
                <AdvancedAnalyticsView data={advancedData} />
              )}
            </Box>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
