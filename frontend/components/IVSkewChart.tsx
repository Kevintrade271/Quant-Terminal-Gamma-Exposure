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
import type { IVSkewResponse } from '@/lib/types';
import ExportChartButton from './ExportChartButton';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface IVSkewChartProps {
  data: IVSkewResponse;
}

export default function IVSkewChart({ data }: IVSkewChartProps) {
  const chartRef = useRef<any>(null);
  const [showExtraMetrics, setShowExtraMetrics] = useState({
    putCallRatio: true, // Show Put/Call ratio panel by default
  });

  if (!data || !data.data || data.data.length === 0) {
    return (
      <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
        <CardContent>
          <Typography variant="h6" color="text.secondary">
            Esperando datos de IV Skew...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const expirations = [...new Set(data.data.map((d) => d.exp))].sort();
  const nearestExp = expirations[0];

  const expData = data.data.filter((d) => d.exp === nearestExp);

  const callsData = expData.filter((d) => d.side === 'calls').sort((a, b) => a.moneyness - b.moneyness);
  const putsData = expData.filter((d) => d.side === 'puts').sort((a, b) => a.moneyness - b.moneyness);

  const chartData = {
    datasets: [
      {
        label: 'Calls IV',
        data: callsData.map((d) => ({ x: d.moneyness, y: d.iv * 100 })),
        borderColor: '#8b5cf6',
        backgroundColor: '#8b5cf6',
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.3,
      },
      {
        label: 'Puts IV',
        data: putsData.map((d) => ({ x: d.moneyness, y: d.iv * 100 })),
        borderColor: '#ef4444',
        backgroundColor: '#ef4444',
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.3,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
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
        text: `IV Skew — ${data.ticker} @ $${data.spot.toFixed(2)} | Exp: ${nearestExp}`,
        color: '#e5e7eb',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            const label = context.dataset.label || '';
            const moneyness = context.parsed.x.toFixed(4);
            const iv = context.parsed.y.toFixed(2);
            return `${label}: ${iv}% (Moneyness: ${moneyness})`;
          },
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: 'Moneyness (Strike / Spot)',
          color: '#e5e7eb',
        },
        grid: {
          color: '#374151',
        },
        ticks: {
          color: '#9ca3af',
          callback: function (value) {
            return (value as number).toFixed(2);
          },
        },
      },
      y: {
        title: {
          display: true,
          text: 'Implied Volatility (%)',
          color: '#e5e7eb',
        },
        grid: {
          color: '#374151',
        },
        ticks: {
          color: '#9ca3af',
          callback: function (value) {
            return `${value}%`;
          },
        },
      },
    },
  };

  // Cálculo de métricas adicionales
  const atmMoneyness = 1.0;
  const atmCall = callsData.find((d) => Math.abs(d.moneyness - atmMoneyness) < 0.05);
  const atmPut = putsData.find((d) => Math.abs(d.moneyness - atmMoneyness) < 0.05);

  const atmCallIV = atmCall ? atmCall.iv * 100 : 0;
  const atmPutIV = atmPut ? atmPut.iv * 100 : 0;

  // Skew = diferencia entre OTM put IV y ATM IV
  const otmPut = putsData.find((d) => d.moneyness < 0.95);
  const putSkew = otmPut ? (otmPut.iv * 100 - atmPutIV) : 0;

  // Diferencia call-put skew
  const callPutDiff = Math.abs(atmCallIV - atmPutIV);

  // Contar oportunidades (donde IV está muy por encima/debajo de ATM)
  const avgCallIV = callsData.reduce((sum, d) => sum + d.iv * 100, 0) / callsData.length;
  const avgPutIV = putsData.reduce((sum, d) => sum + d.iv * 100, 0) / putsData.length;
  const opportunityCount = expData.filter((d) => {
    const iv = d.iv * 100;
    const avg = d.side === 'calls' ? avgCallIV : avgPutIV;
    return Math.abs(iv - avg) > avg * 0.15; // 15% desviación
  }).length;

  const formatPercent = (num: number) => `${num.toFixed(2)}%`;

  // Put/Call Skew Ratio Analysis
  const putCallIVRatio = atmPutIV > 0 ? atmCallIV / atmPutIV : 1;
  const isInverted = putCallIVRatio > 1.1; // Calls more expensive than puts (unusual)
  const isNormal = putCallIVRatio < 0.9; // Puts more expensive (normal)
  const isBalanced = !isInverted && !isNormal;

  // Extreme ratios suggest potential reversals
  const isExtremelyInverted = putCallIVRatio > 1.2; // Very unusual - bearish sentiment exhaustion?
  const isExtremelySkewed = putCallIVRatio < 0.75; // Very high put premium - bullish sentiment exhaustion?

  const getSkewSentiment = () => {
    if (isExtremelyInverted) return { text: 'ALERTA: Calls Extremadamente Caros', color: '#dc2626', signal: '⚠️ Posible Reversión Bajista' };
    if (isExtremelySkewed) return { text: 'ALERTA: Puts Extremadamente Caros', color: '#dc2626', signal: '⚠️ Posible Reversión Alcista' };
    if (isInverted) return { text: 'Sesgo Inusual: Calls > Puts', color: '#f59e0b', signal: '👀 Monitorear' };
    if (isNormal) return { text: 'Sesgo Normal: Puts > Calls', color: '#10b981', signal: '✅ Normal' };
    return { text: 'Sesgo Balanceado', color: '#6b7280', signal: '➡️ Neutral' };
  };

  const skewSentiment = getSkewSentiment();

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <Typography variant="body2" sx={{ color: '#9ca3af' }}>
              Put Skew: {putSkew > 0 ? '+' : ''}{formatPercent(putSkew)} | C-P Diff: {formatPercent(callPutDiff)} | Oportunidades: {opportunityCount}
            </Typography>
            <FormControlLabel
              control={
                <Checkbox
                  checked={showExtraMetrics.putCallRatio}
                  onChange={(e) =>
                    setShowExtraMetrics({ ...showExtraMetrics, putCallRatio: e.target.checked })
                  }
                  size="small"
                  sx={{
                    color: '#8b5cf6',
                    '&.Mui-checked': { color: '#8b5cf6' }
                  }}
                />
              }
              label={<Typography sx={{ fontSize: '0.875rem', fontWeight: 500 }}>Análisis Put/Call Ratio</Typography>}
            />
          </Box>
          <ExportChartButton chartRef={chartRef} filename={`${data.ticker}_IVSkew`} />
        </Box>

        <Box sx={{ height: 500 }}>
          <Line ref={chartRef} data={chartData} options={options} />
        </Box>

        {showExtraMetrics.putCallRatio && (
          <Card sx={{ bgcolor: '#1f2937', mt: 3, p: 2, border: `2px solid ${skewSentiment.color}` }}>
            <Typography variant="h6" sx={{ color: skewSentiment.color, fontWeight: 700, mb: 1 }}>
              {skewSentiment.signal}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <Typography variant="body1" sx={{ color: '#e5e7eb', fontWeight: 600 }}>
                Put/Call IV Ratio:
              </Typography>
              <Typography variant="h5" sx={{ color: skewSentiment.color, fontWeight: 700 }}>
                {putCallIVRatio.toFixed(3)}
              </Typography>
              <Typography variant="body2" sx={{ color: '#9ca3af' }}>
                ({skewSentiment.text})
              </Typography>
            </Box>

            {/* Visual Gauge */}
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>Puts Caros</Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>Balanceado</Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>Calls Caros</Typography>
              </Box>
              <Box sx={{ position: 'relative', height: 30, bgcolor: '#374151', borderRadius: 1, overflow: 'hidden' }}>
                {/* Background gradient */}
                <Box sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'linear-gradient(to right, #10b981 0%, #6b7280 50%, #f59e0b 85%, #dc2626 100%)'
                }} />
                {/* Indicator marker */}
                <Box sx={{
                  position: 'absolute',
                  top: '50%',
                  left: `${Math.min(Math.max((putCallIVRatio - 0.5) / 1.0 * 100, 0), 100)}%`,
                  transform: 'translate(-50%, -50%)',
                  width: 4,
                  height: '100%',
                  bgcolor: '#fff',
                  boxShadow: '0 0 10px rgba(255,255,255,0.8)'
                }} />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>0.50</Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>1.00</Typography>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>1.50</Typography>
              </Box>
            </Box>

            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
              <Box>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>ATM Call IV</Typography>
                <Typography variant="body1" sx={{ color: '#8b5cf6', fontWeight: 600 }}>
                  {formatPercent(atmCallIV)}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" sx={{ color: '#9ca3af' }}>ATM Put IV</Typography>
                <Typography variant="body1" sx={{ color: '#ef4444', fontWeight: 600 }}>
                  {formatPercent(atmPutIV)}
                </Typography>
              </Box>
            </Box>
          </Card>
        )}
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          El IV Skew muestra cómo varía la volatilidad implícita según el strike.
          <br />
          Un skew pronunciado indica miedo direccional del mercado.
        </Typography>
      </CardContent>
    </Card>
  );
}
