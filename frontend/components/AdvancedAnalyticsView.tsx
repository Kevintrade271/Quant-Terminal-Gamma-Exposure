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
    gammaProjections: true, // Show projections by default
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

  // Gamma Profile Projections: Calculate Net GEX at different price levels
  const calculateNetGexAtPrice = (targetPrice: number): number => {
    const { strikes, net_gamma } = net_gamma_profile;
    // Find closest strike to target price
    const closestIndex = strikes.reduce((prevIdx, currStrike, currIdx) => {
      return Math.abs(currStrike - targetPrice) < Math.abs(strikes[prevIdx] - targetPrice) ? currIdx : prevIdx;
    }, 0);
    return net_gamma[closestIndex];
  };

  const projections = [
    { label: '-10%', price: spot * 0.90, percent: -10 },
    { label: '-5%', price: spot * 0.95, percent: -5 },
    { label: '-2%', price: spot * 0.98, percent: -2 },
    { label: 'Spot', price: spot, percent: 0 },
    { label: '+2%', price: spot * 1.02, percent: 2 },
    { label: '+5%', price: spot * 1.05, percent: 5 },
    { label: '+10%', price: spot * 1.10, percent: 10 },
  ].map(proj => ({
    ...proj,
    netGex: calculateNetGexAtPrice(proj.price),
  }));

  // Determine which direction has more resistance/support based on GEX
  const upside5 = projections.find(p => p.percent === 5)?.netGex || 0;
  const downside5 = projections.find(p => p.percent === -5)?.netGex || 0;
  const projectionBias = upside5 > Math.abs(downside5) ? 'Resistencia al alza' : 'Soporte a la baja';

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, gap: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <Typography variant="body2" sx={{ color: '#9ca3af' }}>
            Tendencia: {levelTrend}
            {gammaFlipDistance !== null && ` | Flip: ${gammaFlipDistance > 0 ? '+' : ''}${gammaFlipDistance.toFixed(1)}%`}
            {alerts.length > 0 && ` | 🚨 ${alerts.join(', ')}`}
          </Typography>
          <FormControlLabel
            control={
              <Checkbox
                checked={showExtraMetrics.gammaProjections}
                onChange={(e) =>
                  setShowExtraMetrics({ ...showExtraMetrics, gammaProjections: e.target.checked })
                }
                size="small"
                sx={{
                  color: '#8b5cf6',
                  '&.Mui-checked': { color: '#8b5cf6' }
                }}
              />
            }
            label={<Typography sx={{ fontSize: '0.875rem', fontWeight: 500 }}>Proyecciones GEX</Typography>}
          />
        </Box>
      </Box>

      {showExtraMetrics.gammaProjections && (
        <Card sx={{ bgcolor: '#1f2937', mb: 3, p: 2, border: '2px solid #8b5cf6' }}>
          <Typography variant="h6" sx={{ color: '#8b5cf6', fontWeight: 700, mb: 1 }}>
            📊 Proyecciones de Gamma Profile
          </Typography>
          <Typography variant="body2" sx={{ color: '#9ca3af', mb: 2 }}>
            Net GEX proyectado a diferentes niveles de precio. {projectionBias}.
          </Typography>

          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 1 }}>
            {projections.map((proj, idx) => {
              const isSpot = proj.percent === 0;
              const color = proj.netGex > 0 ? '#10b981' : proj.netGex < 0 ? '#ef4444' : '#6b7280';
              const bgColor = isSpot ? '#fbbf24' : '#374151';

              return (
                <Box
                  key={idx}
                  sx={{
                    bgcolor: bgColor,
                    p: 1.5,
                    borderRadius: 1,
                    textAlign: 'center',
                    border: isSpot ? '2px solid #facc15' : `1px solid ${color}`,
                  }}
                >
                  <Typography variant="caption" sx={{ color: isSpot ? '#1f2937' : '#e5e7eb', fontWeight: 700, display: 'block', mb: 0.5 }}>
                    {proj.label}
                  </Typography>
                  <Typography variant="caption" sx={{ color: isSpot ? '#1f2937' : '#9ca3af', display: 'block', fontSize: '0.7rem' }}>
                    ${proj.price.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" sx={{ color: isSpot ? '#1f2937' : color, fontWeight: 700, mt: 0.5 }}>
                    {formatLargeNumber(Math.abs(proj.netGex))}
                  </Typography>
                  <Typography variant="caption" sx={{ color: isSpot ? '#1f2937' : '#9ca3af', fontSize: '0.65rem' }}>
                    {proj.netGex > 0 ? 'Call' : proj.netGex < 0 ? 'Put' : 'Neutral'}
                  </Typography>
                </Box>
              );
            })}
          </Box>

          <Box sx={{ mt: 2, p: 1.5, bgcolor: '#374151', borderRadius: 1 }}>
            <Typography variant="caption" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
              💡 Interpretación: Un Net GEX positivo alto indica que los dealers amplificarán movimientos alcistas (más volatilidad al subir).
              Un Net GEX negativo alto indica que los dealers frenarán movimientos bajistas (menos volatilidad al bajar).
            </Typography>
          </Box>
        </Card>
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
