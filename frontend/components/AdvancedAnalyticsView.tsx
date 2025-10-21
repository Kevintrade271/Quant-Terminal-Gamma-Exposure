'use client';

import { Card, CardContent, Typography, Box, FormControlLabel, Checkbox, FormGroup } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import type { AdvancedAnalyticsResponse } from '@/lib/types';
import OperationalLevelsPanel from './OperationalLevelsPanel';
import ExportChartButton from './ExportChartButton';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  annotationPlugin
);

interface AdvancedAnalyticsViewProps {
  data: AdvancedAnalyticsResponse;
}

export default function AdvancedAnalyticsView({ data }: AdvancedAnalyticsViewProps) {
  const chartRef = useRef<any>(null);
  const [showExtraMetrics, setShowExtraMetrics] = useState({
    levelTrends: false,
    gammaFlipHistory: false,
    alerts: false,
  });

  if (!data || !data.net_gamma_profile) {
    return (
      <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
        <CardContent>
          <Typography variant="h6" color="text.secondary">
            Esperando datos de analytics avanzados...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const { net_gamma_profile, operational_levels, spot } = data;

  const annotations: any = {};

  if (operational_levels.gamma_pin) {
    annotations.gammaPin = {
      type: 'line',
      xMin: operational_levels.gamma_pin,
      xMax: operational_levels.gamma_pin,
      borderColor: '#fbbf24',
      borderWidth: 3,
      borderDash: [10, 5],
      label: {
        display: true,
        content: `Gamma Pin: $${operational_levels.gamma_pin.toFixed(2)}`,
        position: 'start',
        backgroundColor: '#fbbf24',
        color: '#1f2937',
        font: {
          weight: 'bold',
        },
      },
    };
  }

  if (operational_levels.slip_risk) {
    annotations.slipRisk = {
      type: 'line',
      xMin: operational_levels.slip_risk,
      xMax: operational_levels.slip_risk,
      borderColor: '#ef4444',
      borderWidth: 2,
      borderDash: [5, 5],
      label: {
        display: true,
        content: `Slip Risk: $${operational_levels.slip_risk.toFixed(2)}`,
        position: 'end',
        backgroundColor: '#ef4444',
        color: '#fff',
      },
    };
  }

  if (operational_levels.exhaustion) {
    annotations.exhaustion = {
      type: 'line',
      xMin: operational_levels.exhaustion,
      xMax: operational_levels.exhaustion,
      borderColor: '#8b5cf6',
      borderWidth: 2,
      borderDash: [5, 5],
      label: {
        display: true,
        content: `Exhaustion: $${operational_levels.exhaustion.toFixed(2)}`,
        position: 'end',
        backgroundColor: '#8b5cf6',
        color: '#fff',
      },
    };
  }

  if (operational_levels.convex_hotspot) {
    annotations.convexHotspot = {
      type: 'line',
      xMin: operational_levels.convex_hotspot,
      xMax: operational_levels.convex_hotspot,
      borderColor: '#10b981',
      borderWidth: 2,
      borderDash: [5, 5],
      label: {
        display: true,
        content: `Convex Hotspot: $${operational_levels.convex_hotspot.toFixed(2)}`,
        position: 'end',
        backgroundColor: '#10b981',
        color: '#fff',
      },
    };
  }

  annotations.spot = {
    type: 'line',
    xMin: spot,
    xMax: spot,
    borderColor: '#facc15',
    borderWidth: 3,
    label: {
      display: true,
      content: `Spot: $${spot.toFixed(2)}`,
      position: 'center',
      backgroundColor: '#facc15',
      color: '#1f2937',
      font: {
        weight: 'bold',
      },
    },
  };

  const chartData = {
    labels: net_gamma_profile.strikes.map((s) => s.toFixed(2)),
    datasets: [
      {
        label: 'Net Gamma Profile',
        data: net_gamma_profile.net_gamma,
        borderColor: '#60a5fa',
        backgroundColor: 'rgba(96, 165, 250, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      annotation: {
        annotations,
      },
      legend: {
        position: 'top' as const,
        labels: {
          color: '#e5e7eb',
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: true,
        text: `Net Gamma Profile — ${data.ticker} @ $${spot.toFixed(2)}`,
        color: '#e5e7eb',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function (context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            const strike = net_gamma_profile.strikes[context.dataIndex];
            return `${label}: ${value.toFixed(2)} (Strike: $${strike.toFixed(2)})`;
          },
        },
      },
    },
    scales: {
      x: {
        type: 'category',
        title: {
          display: true,
          text: 'Strike Price',
          color: '#e5e7eb',
        },
        grid: {
          color: '#374151',
        },
        ticks: {
          color: '#9ca3af',
          maxTicksLimit: 15,
        },
      },
      y: {
        title: {
          display: true,
          text: 'Net Gamma Exposure',
          color: '#e5e7eb',
        },
        grid: {
          color: '#374151',
        },
        ticks: {
          color: '#9ca3af',
        },
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
  };

  const totalCallGex = data.gamma_levels.call_wall_gex;
  const totalPutGex = Math.abs(data.gamma_levels.put_wall_gex);
  const netGex = totalCallGex - totalPutGex;
  const gexRatio = totalPutGex > 0 ? (totalCallGex / totalPutGex) : 0;

  const formatLargeNumber = (num: number) => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toFixed(0);
  };

  // Métricas adicionales
  const gammaFlipDistance = data.gamma_levels.gamma_flip ? ((data.gamma_levels.gamma_flip - spot) / spot) * 100 : null;

  // Detectar alertas
  const alerts: string[] = [];
  if (operational_levels.slip_risk && spot > operational_levels.slip_risk * 0.99) {
    alerts.push('Cerca de Slip Risk');
  }
  if (operational_levels.exhaustion && spot < operational_levels.exhaustion * 1.01) {
    alerts.push('Cerca de Exhaustion');
  }
  if (gexRatio < 0.5) {
    alerts.push('Put Wall muy dominante');
  }
  if (gexRatio > 2) {
    alerts.push('Call Wall muy dominante');
  }

  // Tendencia de niveles (simulado - normalmente vendría del backend)
  const levelTrend = gammaFlipDistance && gammaFlipDistance > 0 ? 'Alcista' : 'Bajista';

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center', mb: 2, gap: 2 }}>
        <FormGroup row>
          <FormControlLabel
            control={
              <Checkbox
                checked={showExtraMetrics.levelTrends}
                onChange={(e) =>
                  setShowExtraMetrics({ ...showExtraMetrics, levelTrends: e.target.checked })
                }
                size="small"
              />
            }
            label={`Tendencia: ${levelTrend}`}
          />
          {gammaFlipDistance !== null && (
            <FormControlLabel
              control={
                <Checkbox
                  checked={showExtraMetrics.gammaFlipHistory}
                  onChange={(e) =>
                    setShowExtraMetrics({ ...showExtraMetrics, gammaFlipHistory: e.target.checked })
                  }
                  size="small"
                />
              }
              label={`Flip: ${gammaFlipDistance > 0 ? '+' : ''}${gammaFlipDistance.toFixed(1)}%`}
            />
          )}
          <FormControlLabel
            control={
              <Checkbox
                checked={showExtraMetrics.alerts}
                onChange={(e) =>
                  setShowExtraMetrics({ ...showExtraMetrics, alerts: e.target.checked })
                }
                size="small"
              />
            }
            label={`Alertas: ${alerts.length}`}
          />
        </FormGroup>
      </Box>

      {showExtraMetrics.alerts && alerts.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" sx={{ color: '#ef4444', fontWeight: 600 }}>
            🚨 {alerts.join(' | ')}
          </Typography>
        </Box>
      )}

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 2, mb: 3 }}>
        <Card sx={{ bgcolor: 'background.paper' }}>
          <CardContent>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
              Call Wall GEX
            </Typography>
            <Typography variant="h5" color="primary.main" sx={{ fontWeight: 700, mt: 0.5 }}>
              {formatLargeNumber(totalCallGex)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              @ ${data.gamma_levels.call_wall.toFixed(2)}
            </Typography>
          </CardContent>
        </Card>

        <Card sx={{ bgcolor: 'background.paper' }}>
          <CardContent>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
              Put Wall GEX
            </Typography>
            <Typography variant="h5" color="error.main" sx={{ fontWeight: 700, mt: 0.5 }}>
              {formatLargeNumber(totalPutGex)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              @ ${data.gamma_levels.put_wall.toFixed(2)}
            </Typography>
          </CardContent>
        </Card>

        <Card sx={{ bgcolor: 'background.paper' }}>
          <CardContent>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
              Net GEX
            </Typography>
            <Typography
              variant="h5"
              sx={{
                fontWeight: 700,
                mt: 0.5,
                color: netGex > 0 ? 'primary.main' : 'error.main',
              }}
            >
              {netGex > 0 ? '+' : ''}{formatLargeNumber(netGex)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {netGex > 0 ? 'Call dominante' : 'Put dominante'}
            </Typography>
          </CardContent>
        </Card>

        <Card sx={{ bgcolor: 'background.paper' }}>
          <CardContent>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
              Call/Put Ratio
            </Typography>
            <Typography variant="h5" color="text.primary" sx={{ fontWeight: 700, mt: 0.5 }}>
              {gexRatio.toFixed(2)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {gexRatio > 1 ? 'Bullish' : gexRatio < 1 ? 'Bearish' : 'Neutral'}
            </Typography>
          </CardContent>
        </Card>
      </Box>

      <Box sx={{ mb: 3 }}>
        <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
              <ExportChartButton chartRef={chartRef} filename={`${data.ticker}_NetGammaProfile`} />
            </Box>
            <Box sx={{ height: 500 }}>
              <Line ref={chartRef} data={chartData} options={options} />
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
              El Net Gamma Profile muestra la exposición gamma neta continua. Las zonas positivas indican donde los dealers amplifican movimientos,
              las negativas donde los suavizan.
            </Typography>
          </CardContent>
        </Card>
      </Box>

      <Box>
        <OperationalLevelsPanel levels={operational_levels} spot={spot} ticker={data.ticker} />
      </Box>
    </Box>
  );
}
