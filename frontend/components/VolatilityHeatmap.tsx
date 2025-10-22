'use client';

import { Card, CardContent, Typography, Box, FormControlLabel, Checkbox, FormGroup } from '@mui/material';
import { useRef, useState } from 'react';
import type { VolatilityResponse } from '@/lib/types';
import ExportChartButton from './ExportChartButton';
import IVTermStructure from './IVTermStructure';

interface VolatilityHeatmapProps {
  data: VolatilityResponse;
}

export default function VolatilityHeatmap({ data }: VolatilityHeatmapProps) {
  const heatmapRef = useRef<HTMLDivElement>(null);
  const [showExtraMetrics, setShowExtraMetrics] = useState({
    ivTermStructure: true,
    extremeVolatility: false,
    averageComparison: false,
  });

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
  const strikes = Array.from(allStrikes).sort((a, b) => parseFloat(a) - parseFloat(b));

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

  // Cálculo de métricas adicionales
  const allValues: number[] = [];
  expirations.forEach((exp) => {
    strikes.forEach((strike) => {
      const value = data.matrix[exp]?.[strike];
      if (value !== null && value !== undefined) {
        allValues.push(value * 100);
      }
    });
  });

  const avgIV = allValues.length > 0 ? allValues.reduce((a, b) => a + b, 0) / allValues.length : 0;
  const maxIV = allValues.length > 0 ? Math.max(...allValues) : 0;
  const minIV = allValues.length > 0 ? Math.min(...allValues) : 0;

  // Strikes con IV inusual (> 1.5 desviaciones estándar de la media)
  const stdDev = Math.sqrt(
    allValues.reduce((sum, val) => sum + Math.pow(val - avgIV, 2), 0) / allValues.length
  );
  const unusualThreshold = avgIV + 1.5 * stdDev;
  const extremeCount = allValues.filter((v) => v > unusualThreshold).length;

  const formatPercent = (num: number) => `${num.toFixed(2)}%`;

  return (
    <>
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6" color="text.primary">
              {data.ticker} Volatility | Spot: ${data.spot.toFixed(2)} | VIX Z: {data.vix_zscore >= 0 ? '+' : ''}
              {data.vix_zscore.toFixed(2)} ({colorTheme})
            </Typography>
            <FormGroup row>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showExtraMetrics.extremeVolatility}
                    onChange={(e) =>
                      setShowExtraMetrics({ ...showExtraMetrics, extremeVolatility: e.target.checked })
                    }
                    size="small"
                  />
                }
                label={`Max IV: ${formatPercent(maxIV)}`}
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showExtraMetrics.averageComparison}
                    onChange={(e) =>
                      setShowExtraMetrics({ ...showExtraMetrics, averageComparison: e.target.checked })
                    }
                    size="small"
                  />
                }
                label={`Avg: ${formatPercent(avgIV)}`}
              />
              <FormControlLabel
                control={
                  <Checkbox
                    checked={showExtraMetrics.unusualStrikes}
                    onChange={(e) =>
                      setShowExtraMetrics({ ...showExtraMetrics, unusualStrikes: e.target.checked })
                    }
                    size="small"
                  />
                }
                label={`Inusuales: ${extremeCount}`}
              />
            </FormGroup>
          </Box>
          <ExportChartButton elementRef={heatmapRef} filename={`${data.ticker}_Volatility`} />
        </Box>

        <Box ref={heatmapRef} sx={{ overflowX: 'auto', mt: 2 }}>
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
                    let bgColor =
                      value !== null && value !== undefined
                        ? getBackgroundColor(value * 100, data.vix_zscore)
                        : '#1f2937';
                    const textColor = value !== null && value !== undefined && value * 100 > 30 ? '#111827' : '#f9fafb';

                    // Highlight based on checkboxes
                    const ivValue = value !== null && value !== undefined ? value * 100 : 0;
                    const isExtreme = showExtraMetrics.extremeVolatility && ivValue >= maxIV * 0.95;
                    const isUnusual = showExtraMetrics.unusualStrikes && ivValue > unusualThreshold;

                    if (isExtreme) {
                      bgColor = '#dc2626';
                    } else if (isUnusual) {
                      bgColor = '#f59e0b';
                    }

                    return (
                      <td
                        key={strike}
                        style={{
                          border: '1px solid #374151',
                          padding: '8px',
                          textAlign: 'center',
                          backgroundColor: bgColor,
                          color: textColor,
                          fontWeight: isExtreme || isUnusual ? 700 : 500,
                          boxShadow: isExtreme ? '0 0 10px rgba(220, 38, 38, 0.5)' : isUnusual ? '0 0 8px rgba(245, 158, 11, 0.5)' : 'none',
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
    {showExtraMetrics.ivTermStructure && <IVTermStructure data={data} />}
    </>
  );
}
