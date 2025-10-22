'use client';

import { Card, CardContent, Typography, Box, FormControlLabel, Checkbox, FormGroup, LinearProgress } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import type { GreekDataPoint } from '@/lib/types';
import ExportChartButton from './ExportChartButton';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  annotationPlugin
);

interface GexCharmChartProps {
  data: GreekDataPoint[];
  spot: number;
  ticker: string;
  valueType: 'GEX' | 'CHARM';
  gammaFlip?: number | null;
  callWall?: number | null;
  putWall?: number | null;
  zoomPct?: number;
}

export default function GexCharmChart({
  data,
  spot,
  ticker,
  valueType,
  gammaFlip,
  callWall,
  putWall,
  zoomPct = 0.02,
}: GexCharmChartProps) {
  const chartRef = useRef<any>(null);
  const [showEnhancedMetrics, setShowEnhancedMetrics] = useState({
    netGexOverlay: false,
    gexSkew: false,
    supportResistanceZones: false,
  });

  if (!data || data.length === 0) {
    return (
      <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
        <CardContent>
          <Typography variant="h6" color="text.secondary">
            Sin datos para {valueType}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const exps = [...new Set(data.map((d) => d.exp))].sort();
  const nearestExp = exps[0];

  const filteredData = data.filter((d) => {
    const kMin = spot * (1 - zoomPct);
    const kMax = spot * (1 + zoomPct);
    return d.exp === nearestExp && d.K >= kMin && d.K <= kMax;
  });

  const strikes = [...new Set(filteredData.map((d) => d.K))].sort((a, b) => a - b);

  const callsData = strikes.map((strike) => {
    const point = filteredData.find((d) => d.K === strike && d.side === 'C');
    return point ? point[valueType] : 0;
  });

  const putsData = strikes.map((strike) => {
    const point = filteredData.find((d) => d.K === strike && d.side === 'P');
    return point ? -point[valueType] : 0;
  });

  // Calcular Net GEX (Calls - Puts absolutos)
  const netGexData = strikes.map((strike, idx) => {
    const callPoint = filteredData.find((d) => d.K === strike && d.side === 'C');
    const putPoint = filteredData.find((d) => d.K === strike && d.side === 'P');
    const callVal = callPoint ? callPoint[valueType] : 0;
    const putVal = putPoint ? putPoint[valueType] : 0;
    return callVal - putVal;
  });

  // Calcular GEX Skew
  const calculateGexSkew = () => {
    let weightedSum = 0;
    let totalGex = 0;

    filteredData.forEach((point) => {
      const distance = point.K - spot;
      const gexValue = Math.abs(point[valueType]);
      weightedSum += gexValue * distance;
      totalGex += gexValue;
    });

    return totalGex > 0 ? (weightedSum / totalGex) / spot * 100 : 0;
  };

  const gexSkew = calculateGexSkew();

  // Identificar zonas de soporte/resistencia
  const identifySupportResistanceZones = () => {
    const zones: { start: number; end: number; type: 'support' | 'resistance'; strength: number }[] = [];
    const threshold = Math.max(...netGexData.map(Math.abs)) * 0.3;

    let currentZone: { start: number; type: 'support' | 'resistance'; sum: number } | null = null;

    netGexData.forEach((netGex, idx) => {
      if (Math.abs(netGex) > threshold) {
        const type = netGex > 0 ? 'support' : 'resistance';

        if (currentZone && currentZone.type === type) {
          currentZone.sum += Math.abs(netGex);
        } else {
          if (currentZone) {
            zones.push({
              start: currentZone.start,
              end: idx - 1,
              type: currentZone.type,
              strength: currentZone.sum
            });
          }
          currentZone = { start: idx, type, sum: Math.abs(netGex) };
        }
      } else {
        if (currentZone) {
          zones.push({
            start: currentZone.start,
            end: idx - 1,
            type: currentZone.type,
            strength: currentZone.sum
          });
          currentZone = null;
        }
      }
    });

    if (currentZone) {
      zones.push({
        start: currentZone.start,
        end: netGexData.length - 1,
        type: currentZone.type,
        strength: currentZone.sum
      });
    }

    return zones;
  };

  const supportResistanceZones = identifySupportResistanceZones();

  const datasets: any[] = [
    {
      label: 'Puts',
      data: putsData,
      backgroundColor: '#ef4444',
      borderColor: '#dc2626',
      borderWidth: 1,
      type: 'bar' as const,
    },
    {
      label: 'Calls',
      data: callsData,
      backgroundColor: '#8b5cf6',
      borderColor: '#7c3aed',
      borderWidth: 1,
      type: 'bar' as const,
    },
  ];

  if (showEnhancedMetrics.netGexOverlay && valueType === 'GEX') {
    datasets.push({
      label: 'Net GEX',
      data: netGexData,
      type: 'line' as const,
      borderColor: (context: any) => {
        const value = context.raw;
        return value >= 0 ? '#10b981' : '#f59e0b';
      },
      segment: {
        borderColor: (context: any) => {
          const current = context.p1.parsed.y;
          return current >= 0 ? '#10b981' : '#f59e0b';
        }
      },
      backgroundColor: 'transparent',
      borderWidth: 3,
      pointRadius: 4,
      pointBackgroundColor: (context: any) => {
        const value = context.raw;
        return value >= 0 ? '#10b981' : '#f59e0b';
      },
      pointBorderColor: '#fff',
      pointBorderWidth: 2,
      yAxisID: 'y1',
      tension: 0.2,
    });
  }

  const chartData = {
    labels: strikes.map((s) => s.toFixed(2)),
    datasets,
  };

  const annotations: any = {};

  // Agregar zonas de soporte/resistencia
  if (showEnhancedMetrics.supportResistanceZones && valueType === 'GEX') {
    supportResistanceZones.forEach((zone, idx) => {
      const color = zone.type === 'support' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
      const borderColor = zone.type === 'support' ? '#10b981' : '#ef4444';

      annotations[`zone_${idx}`] = {
        type: 'box',
        xMin: zone.start - 0.5,
        xMax: zone.end + 0.5,
        backgroundColor: color,
        borderColor: borderColor,
        borderWidth: 1,
        borderDash: [3, 3],
        label: {
          display: true,
          content: zone.type === 'support' ? '🛡️ Support' : '⚡ Resistance',
          position: { x: 'center', y: 'start' },
          backgroundColor: borderColor,
          color: '#fff',
          font: { size: 10, weight: 'bold' },
          padding: 4,
        },
      };
    });
  }

  if (gammaFlip && valueType === 'GEX') {
    const flipIndex = strikes.findIndex(s => Math.abs(s - gammaFlip) < 0.5);
    if (flipIndex !== -1) {
      annotations.gammaFlip = {
        type: 'line',
        xMin: flipIndex,
        xMax: flipIndex,
        borderColor: '#fbbf24',
        borderWidth: 3,
        borderDash: [10, 5],
        label: {
          display: true,
          content: `Gamma Flip: $${gammaFlip.toFixed(2)}`,
          position: 'start',
          backgroundColor: '#fbbf24',
          color: '#1f2937',
          font: {
            weight: 'bold',
          },
        },
      };
    }
  }

  if (callWall && valueType === 'GEX') {
    const wallIndex = strikes.findIndex(s => Math.abs(s - callWall) < 0.5);
    if (wallIndex !== -1) {
      annotations.callWall = {
        type: 'line',
        xMin: wallIndex,
        xMax: wallIndex,
        borderColor: '#8b5cf6',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          display: true,
          content: `Call Wall: $${callWall.toFixed(2)}`,
          position: 'end',
          backgroundColor: '#8b5cf6',
          color: '#fff',
        },
      };
    }
  }

  if (putWall && valueType === 'GEX') {
    const wallIndex = strikes.findIndex(s => Math.abs(s - putWall) < 0.5);
    if (wallIndex !== -1) {
      annotations.putWall = {
        type: 'line',
        xMin: wallIndex,
        xMax: wallIndex,
        borderColor: '#ef4444',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          display: true,
          content: `Put Wall: $${putWall.toFixed(2)}`,
          position: 'end',
          backgroundColor: '#ef4444',
          color: '#fff',
        },
      };
    }
  }

  const options: ChartOptions<'bar'> = {
    indexAxis: 'x' as const,
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
        text: `${valueType} Profile — ${ticker} @ $${spot.toFixed(2)} | Exp: ${nearestExp}`,
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
            const value = context.parsed.y;
            const absValue = Math.abs(value);
            const formatted = absValue >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : value.toFixed(2);
            return `${label}: ${formatted}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: '#374151',
        },
        ticks: {
          color: '#9ca3af',
        },
        title: {
          display: true,
          text: 'Strike',
          color: '#e5e7eb',
        },
      },
      y: {
        stacked: false,
        grid: {
          color: '#374151',
        },
        ticks: {
          color: '#9ca3af',
          callback: (value: any) => {
            const val = Number(value);
            if (Math.abs(val) >= 1e6) return `${(val / 1e6).toFixed(1)}M`;
            if (Math.abs(val) >= 1e3) return `${(val / 1e3).toFixed(1)}K`;
            return val.toFixed(0);
          }
        },
        title: {
          display: true,
          text: `${valueType} Exposure`,
          color: '#e5e7eb',
        },
      },
      y1: {
        type: 'linear' as const,
        display: showEnhancedMetrics.netGexOverlay && valueType === 'GEX',
        position: 'right' as const,
        grid: {
          display: false,
        },
        ticks: {
          color: '#10b981',
          callback: (value: any) => {
            const val = Number(value);
            if (Math.abs(val) >= 1e6) return `${(val / 1e6).toFixed(1)}M`;
            if (Math.abs(val) >= 1e3) return `${(val / 1e3).toFixed(1)}K`;
            return val.toFixed(0);
          }
        },
        title: {
          display: true,
          text: 'Net GEX',
          color: '#10b981',
        },
      },
    },
  };

  const getSkewColor = (skew: number) => {
    if (skew > 1.5) return '#10b981';
    if (skew > 0.5) return '#84cc16';
    if (skew > -0.5) return '#f59e0b';
    if (skew > -1.5) return '#f97316';
    return '#ef4444';
  };

  const getSkewLabel = (skew: number) => {
    if (skew > 1.5) return 'Muy Bullish';
    if (skew > 0.5) return 'Bullish';
    if (skew > -0.5) return 'Neutral';
    if (skew > -1.5) return 'Bearish';
    return 'Muy Bearish';
  };

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <FormGroup row>
            {valueType === 'GEX' && (
              <>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={showEnhancedMetrics.netGexOverlay}
                      onChange={(e) =>
                        setShowEnhancedMetrics({ ...showEnhancedMetrics, netGexOverlay: e.target.checked })
                      }
                      size="small"
                      sx={{
                        color: '#10b981',
                        '&.Mui-checked': { color: '#10b981' }
                      }}
                    />
                  }
                  label={<Typography sx={{ fontSize: '0.875rem', fontWeight: 500 }}>Net GEX (línea)</Typography>}
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={showEnhancedMetrics.gexSkew}
                      onChange={(e) =>
                        setShowEnhancedMetrics({ ...showEnhancedMetrics, gexSkew: e.target.checked })
                      }
                      size="small"
                      sx={{
                        color: '#f59e0b',
                        '&.Mui-checked': { color: '#f59e0b' }
                      }}
                    />
                  }
                  label={<Typography sx={{ fontSize: '0.875rem', fontWeight: 500 }}>GEX Skew</Typography>}
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={showEnhancedMetrics.supportResistanceZones}
                      onChange={(e) =>
                        setShowEnhancedMetrics({ ...showEnhancedMetrics, supportResistanceZones: e.target.checked })
                      }
                      size="small"
                      sx={{
                        color: '#8b5cf6',
                        '&.Mui-checked': { color: '#8b5cf6' }
                      }}
                    />
                  }
                  label={<Typography sx={{ fontSize: '0.875rem', fontWeight: 500 }}>Zonas S/R</Typography>}
                />
              </>
            )}
          </FormGroup>
          <ExportChartButton chartRef={chartRef} filename={`${ticker}_${valueType}`} />
        </Box>

        {showEnhancedMetrics.gexSkew && valueType === 'GEX' && (
          <Box sx={{ mb: 2, p: 2, bgcolor: 'background.default', borderRadius: 2 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontWeight: 600 }}>
              GEX Skew Indicator
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="caption" color="text.secondary" sx={{ minWidth: 80 }}>
                Bearish
              </Typography>
              <Box sx={{ flex: 1, position: 'relative' }}>
                <LinearProgress
                  variant="determinate"
                  value={50}
                  sx={{
                    height: 24,
                    borderRadius: 2,
                    bgcolor: '#374151',
                    '& .MuiLinearProgress-bar': {
                      bgcolor: 'transparent'
                    }
                  }}
                />
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Box
                    sx={{
                      position: 'absolute',
                      left: `${Math.min(Math.max((gexSkew + 3) / 6 * 100, 0), 100)}%`,
                      transform: 'translateX(-50%)',
                      width: 3,
                      height: 28,
                      bgcolor: getSkewColor(gexSkew),
                      borderRadius: 1,
                      boxShadow: `0 0 10px ${getSkewColor(gexSkew)}`,
                    }}
                  />
                  <Typography
                    variant="caption"
                    sx={{
                      position: 'absolute',
                      color: '#fff',
                      fontWeight: 700,
                      fontSize: '0.75rem',
                      textShadow: '0 1px 3px rgba(0,0,0,0.5)'
                    }}
                  >
                    {gexSkew.toFixed(2)}% • {getSkewLabel(gexSkew)}
                  </Typography>
                </Box>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ minWidth: 80, textAlign: 'right' }}>
                Bullish
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1, px: 1 }}>
              <Typography variant="caption" sx={{ color: '#ef4444' }}>-3%</Typography>
              <Typography variant="caption" sx={{ color: '#9ca3af' }}>0%</Typography>
              <Typography variant="caption" sx={{ color: '#10b981' }}>+3%</Typography>
            </Box>
          </Box>
        )}

        <Box sx={{ height: 600 }}>
          <Bar ref={chartRef} data={chartData} options={options} />
        </Box>
      </CardContent>
    </Card>
  );
}
