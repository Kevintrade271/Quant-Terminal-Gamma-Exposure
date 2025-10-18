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
} from '@mui/material';
import { theme } from '@/lib/theme';
import { fetchGreeks, fetchVolatility, fetchStatus } from '@/lib/api';
import type { GreeksResponse, VolatilityResponse, StatusResponse, TabValue } from '@/lib/types';
import GexCharmChart from '@/components/GexCharmChart';
import VolatilityHeatmap from '@/components/VolatilityHeatmap';

const TICKER = 'SPY';
const REFRESH_INTERVAL = 5 * 60 * 1000;

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabValue>('gex-charm');
  const [greeksData, setGreeksData] = useState<GreeksResponse | null>(null);
  const [volatilityData, setVolatilityData] = useState<VolatilityResponse | null>(null);
  const [statusData, setStatusData] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const loadData = async () => {
    try {
      setError(null);

      if (activeTab === 'gex-charm') {
        const [greeks, status] = await Promise.all([
          fetchGreeks(TICKER),
          fetchStatus(TICKER),
        ]);
        setGreeksData(greeks);
        setStatusData(status);
      } else {
        const [volatility, status] = await Promise.all([
          fetchVolatility(TICKER),
          fetchStatus(TICKER),
        ]);
        setVolatilityData(volatility);
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
  }, [activeTab]);

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
          <Typography
            variant="h3"
            component="h1"
            gutterBottom
            sx={{
              textAlign: 'center',
              color: 'text.primary',
              fontWeight: 700,
              mb: 2,
            }}
          >
            🎯 Quant Terminal - Dashboard Unificado
          </Typography>

          {statusData && (
            <Card sx={{ mb: 3, bgcolor: 'background.paper' }}>
              <CardContent>
                <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center' }}>
                  ✅ Última Actualización:{' '}
                  {lastUpdate ? lastUpdate.toLocaleTimeString('es-ES') : 'Cargando...'} | Spot: $
                  {statusData.spot.toFixed(2)} | VIX: {statusData.vix_current.toFixed(2)} (Z:{' '}
                  {statusData.vix_zscore >= 0 ? '+' : ''}
                  {statusData.vix_zscore.toFixed(2)})
                </Typography>
              </CardContent>
            </Card>
          )}

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
            </Box>
          )}
        </Container>
      </Box>
    </ThemeProvider>
  );
}
