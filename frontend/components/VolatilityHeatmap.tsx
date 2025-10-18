'use client';

import { Card, CardContent, Typography, Box } from '@mui/material';
import type { VolatilityResponse } from '@/lib/types';

interface VolatilityHeatmapProps {
  data: VolatilityResponse;
}

export default function VolatilityHeatmap({ data }: VolatilityHeatmapProps) {
  if (!data || !data.matrix || Object.keys(data.matrix).length === 0) {
    return (
      <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
        <CardContent>
          <Typography variant="h6" color="text.secondary">
            Esperando datos de volatilidad...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const expirations = Object.keys(data.matrix).sort();
  const allStrikes = new Set<string>();
  expirations.forEach((exp) => {
    Object.keys(data.matrix[exp]).forEach((strike) => allStrikes.add(strike));
  });
  const strikes = Array.from(allStrikes)
    .map(Number)
    .sort((a, b) => a - b)
    .map(String);

  const getColorScale = (vixZscore: number) => {
    if (vixZscore > 0.5) return 'Greens';
    if (vixZscore < -0.5) return 'Reds';
    return 'Greys';
  };

  const getColorTheme = (vixZscore: number) => {
    if (vixZscore > 0.5) return 'Miedo/Oportunidad';
    if (vixZscore < -0.5) return 'Complacencia/Peligro';
    return 'Neutral';
  };

  const getBackgroundColor = (value: number, vixZscore: number) => {
    const intensity = Math.min(value / 50, 1);

    if (vixZscore > 0.5) {
      const green = Math.floor(100 + intensity * 155);
      return `rgb(0, ${green}, 0)`;
    } else if (vixZscore < -0.5) {
      const red = Math.floor(100 + intensity * 155);
      return `rgb(${red}, 0, 0)`;
    } else {
      const gray = Math.floor(100 + intensity * 100);
      return `rgb(${gray}, ${gray}, ${gray})`;
    }
  };

  const colorTheme = getColorTheme(data.vix_zscore);

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom color="text.primary">
          {data.ticker} Volatility | Spot: ${data.spot.toFixed(2)} | VIX Z: {data.vix_zscore >= 0 ? '+' : ''}
          {data.vix_zscore.toFixed(2)} ({colorTheme})
        </Typography>

        <Box sx={{ overflowX: 'auto', mt: 2 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead>
              <tr>
                <th
                  style={{
                    border: '1px solid #374151',
                    padding: '8px',
                    backgroundColor: '#1f2937',
                    color: '#e5e7eb',
                    position: 'sticky',
                    left: 0,
                    zIndex: 10,
                  }}
                >
                  Exp / Strike
                </th>
                {strikes.map((strike) => (
                  <th
                    key={strike}
                    style={{
                      border: '1px solid #374151',
                      padding: '8px',
                      backgroundColor: '#1f2937',
                      color: '#e5e7eb',
                      minWidth: '80px',
                    }}
                  >
                    ${parseFloat(strike).toFixed(2)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {expirations.map((exp) => (
                <tr key={exp}>
                  <td
                    style={{
                      border: '1px solid #374151',
                      padding: '8px',
                      backgroundColor: '#1f2937',
                      color: '#e5e7eb',
                      fontWeight: 'bold',
                      position: 'sticky',
                      left: 0,
                      zIndex: 5,
                    }}
                  >
                    {exp}
                  </td>
                  {strikes.map((strike) => {
                    const value = data.matrix[exp]?.[strike];
                    const displayValue = value !== null && value !== undefined ? (value * 100).toFixed(2) : '-';
                    const bgColor =
                      value !== null && value !== undefined
                        ? getBackgroundColor(value * 100, data.vix_zscore)
                        : '#1f2937';
                    const textColor = value !== null && value !== undefined && value * 100 > 30 ? '#111827' : '#f9fafb';

                    return (
                      <td
                        key={strike}
                        style={{
                          border: '1px solid #374151',
                          padding: '8px',
                          textAlign: 'center',
                          backgroundColor: bgColor,
                          color: textColor,
                          fontWeight: 500,
                        }}
                      >
                        {displayValue}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Box>
      </CardContent>
    </Card>
  );
}
