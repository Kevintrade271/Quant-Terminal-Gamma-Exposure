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
import type { VolatilityResponse } from '@/lib/types';
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

interface IVTermStructureProps {
  data: VolatilityResponse;
}

export default function IVTermStructure({ data }: IVTermStructureProps) {
  const chartRef = useRef<any>(null);

  if (!data || !data.matrix || Object.keys(data.matrix).length === 0) {
    return null;
  }

  const expirations = Object.keys(data.matrix).sort();

  // Calcular ATM IV para cada expiración
  const atmData = expirations.map((exp) => {
    const strikes = Object.keys(data.matrix[exp]).map(Number).sort((a, b) => a - b);
    const atmStrike = strikes.reduce((prev, curr) =>
      Math.abs(curr - data.spot) < Math.abs(prev - data.spot) ? curr : prev
    );
    const atmIV = data.matrix[exp][atmStrike.toString()];
    return {
      exp,
      iv: atmIV ? atmIV * 100 : null,
      dte: calculateDTE(exp)
    };
  }).filter(d => d.iv !== null);

  // Calcular promedio de IVs para toda la superficie
  const avgData = expirations.map((exp) => {
    const ivs = Object.values(data.matrix[exp])
      .filter(v => v !== null && v !== undefined)
      .map(v => v! * 100);
    const avg = ivs.length > 0 ? ivs.reduce((a, b) => a + b, 0) / ivs.length : null;
    return {
      exp,
      iv: avg,
      dte: calculateDTE(exp)
    };
  }).filter(d => d.iv !== null);

  const chartData = {
    labels: atmData.map(d => `${d.dte}d`),
    datasets: [
      {
        label: 'ATM IV',
        data: atmData.map(d => d.iv),
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 3,
        pointRadius: 5,
        pointBackgroundColor: '#8b5cf6',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        tension: 0.3,
        fill: true,
      },
      {
        label: 'Avg IV (superficie completa)',
        data: avgData.map(d => d.iv),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.05)',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 4,
        pointBackgroundColor: '#10b981',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        tension: 0.3,
        fill: true,
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
          font: { size: 12 },
        },
      },
      title: {
        display: true,
        text: 'IV Term Structure (Estructura Temporal de Volatilidad)',
        color: '#e5e7eb',
        font: { size: 14, weight: 'bold' },
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#cbd5e1',
        borderColor: '#334155',
        borderWidth: 1,
        padding: 12,
        callbacks: {
          label: (context) => {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
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
          text: 'Días para Expiración (DTE)',
          color: '#e5e7eb',
        },
      },
      y: {
        grid: {
          color: '#374151',
        },
        ticks: {
          color: '#9ca3af',
          callback: (value: any) => `${value}%`,
        },
        title: {
          display: true,
          text: 'Volatilidad Implícita (%)',
          color: '#e5e7eb',
        },
      },
    },
  };

  const contango = atmData.length >= 2 && atmData[atmData.length - 1].iv! > atmData[0].iv!;
  const backwardation = atmData.length >= 2 && atmData[atmData.length - 1].iv! < atmData[0].iv!;

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2, mt: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 600 }}>
              {contango && '📈 Contango: IV aumenta con el tiempo (bullish de volatilidad)'}
              {backwardation && '📉 Backwardation: IV disminuye con el tiempo (bearish de volatilidad)'}
              {!contango && !backwardation && '➡️ Term Structure plana'}
            </Typography>
          </Box>
          <ExportChartButton chartRef={chartRef} filename={`${data.ticker}_IV_Term_Structure`} />
        </Box>
        <Box sx={{ height: 300 }}>
          <Line ref={chartRef} data={chartData} options={options} />
        </Box>
      </CardContent>
    </Card>
  );
}

function calculateDTE(expiration: string): number {
  const expDate = new Date(expiration);
  const today = new Date();
  const diffTime = expDate.getTime() - today.getTime();
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  return Math.max(0, diffDays);
}
