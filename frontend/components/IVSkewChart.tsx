'use client';

import { Card, CardContent, Typography, Box } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { useRef } from 'react';
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

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
          <ExportChartButton chartRef={chartRef} filename={`${data.ticker}_IVSkew`} />
        </Box>
        <Box sx={{ height: 500 }}>
          <Line ref={chartRef} data={chartData} options={options} />
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
          El IV Skew muestra cómo varía la volatilidad implícita según el strike.
          <br />
          Un skew pronunciado indica miedo direccional del mercado.
        </Typography>
      </CardContent>
    </Card>
  );
}
