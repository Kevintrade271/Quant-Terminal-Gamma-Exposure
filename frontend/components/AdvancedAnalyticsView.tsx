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

  return (
    <Box>
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
              El Net Gamma Profile muestra la exposición gamma neta continua. Las zonas positivas amplifican movimientos,
              las negativas los suavizan.
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
