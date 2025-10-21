'use client';

import { Card, CardContent, Typography, Box, FormControlLabel, Checkbox, FormGroup } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
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
  const [showExtraMetrics, setShowExtraMetrics] = useState({
    totalAboveSpot: false,
    totalBelowSpot: false,
    callPutRatio: false,
    distanceToFlip: false,
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

  const chartData = {
    labels: strikes.map((s) => s.toFixed(2)),
    datasets: [
      {
        label: 'Puts',
        data: putsData,
        backgroundColor: '#ef4444',
        borderColor: '#dc2626',
        borderWidth: 1,
      },
      {
        label: 'Calls',
        data: callsData,
        backgroundColor: '#8b5cf6',
        borderColor: '#7c3aed',
        borderWidth: 1,
      },
    ],
  };

  const annotations: any = {};

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
            return `${label}: ${value.toFixed(2)}`;
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
        },
        title: {
          display: true,
          text: `${valueType} Exposure`,
          color: '#e5e7eb',
        },
      },
    },
  };

  // Cálculo de métricas adicionales
  const totalAboveSpot = filteredData
    .filter((d) => d.K > spot)
    .reduce((sum, d) => sum + d[valueType], 0);

  const totalBelowSpot = filteredData
    .filter((d) => d.K <= spot)
    .reduce((sum, d) => sum + d[valueType], 0);

  const totalCallGex = filteredData
    .filter((d) => d.side === 'C')
    .reduce((sum, d) => sum + d[valueType], 0);

  const totalPutGex = filteredData
    .filter((d) => d.side === 'P')
    .reduce((sum, d) => sum + d[valueType], 0);

  const callPutRatio = totalPutGex !== 0 ? totalCallGex / totalPutGex : 0;

  const distanceToFlip = gammaFlip ? ((gammaFlip - spot) / spot) * 100 : null;

  const formatLargeNumber = (num: number) => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toFixed(0);
  };

  // Agregar anotaciones opcionales a la gráfica
  if (showExtraMetrics.totalAboveSpot) {
    const aboveSpotIndex = strikes.findIndex(s => s > spot);
    if (aboveSpotIndex !== -1) {
      annotations.aboveSpot = {
        type: 'box',
        xMin: aboveSpotIndex - 0.5,
        xMax: strikes.length - 0.5,
        backgroundColor: 'rgba(96, 165, 250, 0.1)',
        borderWidth: 0,
        label: {
          display: true,
          content: `↑ ${formatLargeNumber(totalAboveSpot)}`,
          position: { x: 'end', y: 'start' },
          backgroundColor: '#60a5fa',
          color: '#fff',
          font: { size: 11, weight: 'bold' },
        },
      };
    }
  }

  if (showExtraMetrics.totalBelowSpot) {
    const belowSpotIndex = strikes.findIndex(s => s >= spot);
    if (belowSpotIndex !== -1) {
      annotations.belowSpot = {
        type: 'box',
        xMin: -0.5,
        xMax: belowSpotIndex - 0.5,
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 0,
        label: {
          display: true,
          content: `↓ ${formatLargeNumber(totalBelowSpot)}`,
          position: { x: 'start', y: 'start' },
          backgroundColor: '#ef4444',
          color: '#fff',
          font: { size: 11, weight: 'bold' },
        },
      };
    }
  }

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <FormGroup row>
            <FormControlLabel
              control={
                <Checkbox
                  checked={showExtraMetrics.totalAboveSpot}
                  onChange={(e) =>
                    setShowExtraMetrics({ ...showExtraMetrics, totalAboveSpot: e.target.checked })
                  }
                  size="small"
                />
              }
              label="Zona Superior"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={showExtraMetrics.totalBelowSpot}
                  onChange={(e) =>
                    setShowExtraMetrics({ ...showExtraMetrics, totalBelowSpot: e.target.checked })
                  }
                  size="small"
                />
              }
              label="Zona Inferior"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={showExtraMetrics.callPutRatio}
                  onChange={(e) =>
                    setShowExtraMetrics({ ...showExtraMetrics, callPutRatio: e.target.checked })
                  }
                  size="small"
                />
              }
              label={`Ratio: ${callPutRatio.toFixed(2)}`}
            />
            {gammaFlip && valueType === 'GEX' && distanceToFlip !== null && (
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showExtraMetrics.distanceToFlip}
                    onChange={(e) =>
                      setShowExtraMetrics({ ...showExtraMetrics, distanceToFlip: e.target.checked })
                    }
                    size="small"
                  />
                }
                label={`Flip: ${distanceToFlip > 0 ? '+' : ''}${distanceToFlip.toFixed(1)}%`}
              />
            )}
          </FormGroup>
          <ExportChartButton chartRef={chartRef} filename={`${ticker}_${valueType}`} />
        </Box>

        <Box sx={{ height: 600 }}>
          <Bar ref={chartRef} data={chartData} options={options} />
        </Box>
      </CardContent>
    </Card>
  );
}
