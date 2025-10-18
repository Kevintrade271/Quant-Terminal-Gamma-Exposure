'use client';

import { Card, CardContent, Typography, Box } from '@mui/material';
import { Bar } from 'react-chartjs-2';
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
import type { GreekDataPoint } from '@/lib/types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface GexCharmChartProps {
  data: GreekDataPoint[];
  spot: number;
  ticker: string;
  valueType: 'GEX' | 'CHARM';
  zoomPct?: number;
}

export default function GexCharmChart({
  data,
  spot,
  ticker,
  valueType,
  zoomPct = 0.02,
}: GexCharmChartProps) {
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

  const options: ChartOptions<'bar'> = {
    indexAxis: 'y' as const,
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
            const value = context.parsed.x;
            return `${label}: ${value.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      x: {
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
      y: {
        reverse: true,
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
    },
  };

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Box sx={{ height: 600 }}>
          <Bar data={chartData} options={options} />
        </Box>
      </CardContent>
    </Card>
  );
}
