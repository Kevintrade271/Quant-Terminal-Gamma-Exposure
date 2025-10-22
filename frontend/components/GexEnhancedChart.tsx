'use client';

import { Card, CardContent, Typography, Box } from '@mui/material';
import { Bar, Line } from 'react-chartjs-2';
import { useRef } from 'react';
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
  Filler,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import type { GexEnhancedResponse } from '@/lib/types';
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
  annotationPlugin,
  Filler
);

interface GexEnhancedChartProps {
  data: GexEnhancedResponse;
}

export default function GexEnhancedChart({ data }: GexEnhancedChartProps) {
  const netGexChartRef = useRef<any>(null);
  const volumeChartRef = useRef<any>(null);
  const gammaProfileChartRef = useRef<any>(null);

  if (!data || data.strikes.length === 0) {
    return (
      <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
        <CardContent>
          <Typography variant="h6" color="text.secondary">
            Sin datos para GEX Enhanced
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const { spot, strikes, net_gex_positive, net_gex_negative, absolute_gamma,
          call_volume, put_volume, gamma_flip, gamma_profile, label, window_range } = data;

  const annotations: any = {
    spotLine: {
      type: 'line',
      xMin: strikes.findIndex(s => s >= spot),
      xMax: strikes.findIndex(s => s >= spot),
      borderColor: '#f59e0b',
      borderWidth: 2,
      borderDash: [5, 5],
      label: {
        display: true,
        content: `Spot ${spot.toFixed(0)}`,
        position: 'start',
        backgroundColor: '#f59e0b',
        color: '#fff',
        font: { size: 11, weight: 'bold' }
      }
    }
  };

  if (gamma_flip !== null) {
    annotations.gammaFlipLine = {
      type: 'line',
      xMin: strikes.findIndex(s => s >= gamma_flip),
      xMax: strikes.findIndex(s => s >= gamma_flip),
      borderColor: '#a855f7',
      borderWidth: 2,
      borderDash: [3, 3],
      label: {
        display: true,
        content: `Gamma Flip ${gamma_flip.toFixed(0)}`,
        position: 'end',
        backgroundColor: '#a855f7',
        color: '#fff',
        font: { size: 11, weight: 'bold' }
      }
    };
  }

  const netGexChartData = {
    labels: strikes.map((s) => s.toFixed(2)),
    datasets: [
      {
        label: 'Net GEX (+)',
        data: net_gex_positive,
        backgroundColor: '#3b82f6',
        borderWidth: 0,
        barPercentage: 0.95,
        categoryPercentage: 1,
      },
      {
        label: 'Net GEX (−)',
        data: net_gex_negative,
        backgroundColor: '#ef4444',
        borderWidth: 0,
        barPercentage: 0.95,
        categoryPercentage: 1,
      },
      {
        label: 'Absolute Gamma',
        data: absolute_gamma,
        type: 'line' as const,
        borderColor: '#a78bfa',
        backgroundColor: 'rgba(167, 139, 250, 0.2)',
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        yAxisID: 'y1',
      }
    ]
  };

  const netGexOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      x: {
        stacked: false,
        ticks: {
          color: '#94a3b8',
          maxRotation: 45,
          minRotation: 45,
          autoSkip: true,
          maxTicksLimit: 15
        },
        grid: { color: '#1f2a3a' }
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: {
          display: true,
          text: 'Net GEX ($/1%)',
          color: '#e5e7eb',
          font: { size: 13, weight: 'bold' }
        },
        ticks: {
          color: '#94a3b8',
          callback: (value: any) => {
            const val = Number(value);
            if (Math.abs(val) >= 1e9) return (val / 1e9).toFixed(1) + 'B';
            if (Math.abs(val) >= 1e6) return (val / 1e6).toFixed(1) + 'M';
            if (Math.abs(val) >= 1e3) return (val / 1e3).toFixed(1) + 'K';
            return val.toFixed(0);
          }
        },
        grid: { color: '#1f2a3a' }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'Absolute Gamma',
          color: '#a78bfa',
          font: { size: 13, weight: 'bold' }
        },
        ticks: {
          color: '#94a3b8',
          callback: (value: any) => {
            const val = Number(value);
            if (Math.abs(val) >= 1e9) return (val / 1e9).toFixed(1) + 'B';
            if (Math.abs(val) >= 1e6) return (val / 1e6).toFixed(1) + 'M';
            if (Math.abs(val) >= 1e3) return (val / 1e3).toFixed(1) + 'K';
            return val.toFixed(0);
          }
        },
        grid: { display: false },
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: { color: '#e5e7eb', font: { size: 12 } }
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#cbd5e1',
        borderColor: '#334155',
        borderWidth: 1,
        padding: 12,
        callbacks: {
          label: (context: any) => {
            const val = context.parsed.y;
            const formattedVal = Math.abs(val) >= 1e6
              ? `${(val / 1e6).toFixed(2)}M`
              : val.toFixed(0);
            return `${context.dataset.label}: ${formattedVal}`;
          }
        }
      },
      annotation: { annotations }
    }
  };

  const volumeChartData = {
    labels: strikes.map((s) => s.toFixed(2)),
    datasets: [
      {
        label: 'Call Volume',
        data: call_volume,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.25)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
      },
      {
        label: 'Put Volume',
        data: put_volume,
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
      }
    ]
  };

  const volumeOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      x: {
        ticks: {
          color: '#94a3b8',
          maxRotation: 45,
          minRotation: 45,
          autoSkip: true,
          maxTicksLimit: 15
        },
        grid: { color: '#1f2a3a' }
      },
      y: {
        title: {
          display: true,
          text: 'Volume (smoothed)',
          color: '#e5e7eb',
          font: { size: 13, weight: 'bold' }
        },
        ticks: {
          color: '#94a3b8',
          callback: (value: any) => {
            const val = Number(value);
            if (val >= 1e6) return (val / 1e6).toFixed(1) + 'M';
            if (val >= 1e3) return (val / 1e3).toFixed(1) + 'K';
            return val.toFixed(0);
          }
        },
        grid: { color: '#1f2a3a' }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: { color: '#e5e7eb', font: { size: 12 } }
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#cbd5e1',
        borderColor: '#334155',
        borderWidth: 1,
        padding: 12,
      },
      annotation: { annotations }
    }
  };

  const profileChartData = {
    labels: gamma_profile.price_grid.map((p) => p.toFixed(2)),
    datasets: [
      {
        label: 'Net Gamma Profile',
        data: gamma_profile.profile,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.3,
        pointRadius: 0,
      }
    ]
  };

  const profileOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Price',
          color: '#e5e7eb',
          font: { size: 13, weight: 'bold' }
        },
        ticks: {
          color: '#94a3b8',
          maxRotation: 45,
          minRotation: 45,
          autoSkip: true,
          maxTicksLimit: 12
        },
        grid: { color: '#1f2a3a' }
      },
      y: {
        title: {
          display: true,
          text: 'Total Net GEX',
          color: '#e5e7eb',
          font: { size: 13, weight: 'bold' }
        },
        ticks: {
          color: '#94a3b8',
          callback: (value: any) => {
            const val = Number(value);
            if (Math.abs(val) >= 1e9) return (val / 1e9).toFixed(2) + 'B';
            if (Math.abs(val) >= 1e6) return (val / 1e6).toFixed(2) + 'M';
            return val.toFixed(0);
          }
        },
        grid: { color: '#1f2a3a' }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: { color: '#e5e7eb', font: { size: 12 } }
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#cbd5e1',
        borderColor: '#334155',
        borderWidth: 1,
        padding: 12,
      },
      annotation: {
        annotations: {
          zeroLine: {
            type: 'line',
            yMin: 0,
            yMax: 0,
            borderColor: '#9ca3af',
            borderWidth: 2,
          }
        }
      }
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <Card sx={{ bgcolor: 'background.paper' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ color: 'text.primary', fontWeight: 600 }}>
              📊 {label} — Net GEX & Absolute Gamma (±5% Window)
            </Typography>
            <ExportChartButton chartRef={netGexChartRef} filename={`${data.ticker}_net_gex_enhanced`} />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Ventana: ${window_range.low.toFixed(0)} - ${window_range.high.toFixed(0)}
            {gamma_flip && ` | Gamma Flip: $${gamma_flip.toFixed(2)}`}
          </Typography>
          <Box sx={{ height: '450px' }}>
            <Bar ref={netGexChartRef} data={netGexChartData} options={netGexOptions} />
          </Box>
        </CardContent>
      </Card>

      <Card sx={{ bgcolor: 'background.paper' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ color: 'text.primary', fontWeight: 600 }}>
              📈 Call & Put Volume (Smoothed Areas)
            </Typography>
            <ExportChartButton chartRef={volumeChartRef} filename={`${data.ticker}_volume_smoothed`} />
          </Box>
          <Box sx={{ height: '350px' }}>
            <Line ref={volumeChartRef} data={volumeChartData} options={volumeOptions} />
          </Box>
        </CardContent>
      </Card>

      <Card sx={{ bgcolor: 'background.paper' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ color: 'text.primary', fontWeight: 600 }}>
              🎯 Gamma Profile Curve (Full Range)
            </Typography>
            <ExportChartButton chartRef={gammaProfileChartRef} filename={`${data.ticker}_gamma_profile`} />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Muestra cómo cambia el GEX total a diferentes precios del subyacente
          </Typography>
          <Box sx={{ height: '350px' }}>
            <Line ref={gammaProfileChartRef} data={profileChartData} options={profileOptions} />
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
