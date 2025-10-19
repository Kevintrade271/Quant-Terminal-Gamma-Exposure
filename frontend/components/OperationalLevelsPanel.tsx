'use client';

import { Card, CardContent, Typography, Box, Grid, Chip } from '@mui/material';
import type { OperationalLevels } from '@/lib/types';

interface OperationalLevelsPanelProps {
  levels: OperationalLevels;
  spot: number;
  ticker: string;
}

export default function OperationalLevelsPanel({ levels, spot, ticker }: OperationalLevelsPanelProps) {
  const levelDescriptions = {
    slip_risk: {
      label: 'Slip Risk (Put Wall)',
      description: 'Nivel donde los dealers tienen que vender agresivamente si el precio cae. Soporte fuerte.',
      color: '#ef4444',
    },
    gamma_pin: {
      label: 'Gamma Pin',
      description: 'Zona de máxima concentración gamma cerca del spot. El precio tiende a quedarse aquí.',
      color: '#fbbf24',
    },
    exhaustion: {
      label: 'Exhaustion (Call Wall)',
      description: 'Nivel donde los dealers tienen que comprar agresivamente si el precio sube. Resistencia fuerte.',
      color: '#8b5cf6',
    },
    convex_hotspot: {
      label: 'Convex Hotspot',
      description: 'Zona de máxima curvatura gamma. Movimientos explosivos probables si se cruza.',
      color: '#10b981',
    },
  };

  const calculateDistance = (level: number | null) => {
    if (level === null) return null;
    const distance = ((level - spot) / spot) * 100;
    return distance.toFixed(2);
  };

  return (
    <Card sx={{ bgcolor: 'background.paper', p: 2 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom color="text.primary" sx={{ mb: 3 }}>
          Niveles Operativos - {ticker}
        </Typography>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Estos niveles identifican zonas clave donde el comportamiento del mercado puede cambiar dramáticamente
          debido al hedging de los dealers.
        </Typography>

        <Grid container spacing={2}>
          {(Object.keys(levels) as Array<keyof OperationalLevels>).map((key) => {
            const level = levels[key];
            const info = levelDescriptions[key];
            const distance = calculateDistance(level);

            return (
              <Grid item xs={12} sm={6} key={key}>
                <Box
                  sx={{
                    p: 2,
                    border: '1px solid #374151',
                    borderRadius: 2,
                    borderLeft: `4px solid ${info.color}`,
                    bgcolor: 'rgba(0, 0, 0, 0.2)',
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle1" fontWeight="bold" color="text.primary">
                      {info.label}
                    </Typography>
                    {distance && (
                      <Chip
                        label={`${distance}%`}
                        size="small"
                        sx={{
                          bgcolor: parseFloat(distance) > 0 ? 'rgba(139, 92, 246, 0.3)' : 'rgba(239, 68, 68, 0.3)',
                          color: '#e5e7eb',
                        }}
                      />
                    )}
                  </Box>

                  <Typography variant="h6" color={info.color} sx={{ mb: 1 }}>
                    {level !== null ? `$${level.toFixed(2)}` : 'N/A'}
                  </Typography>

                  <Typography variant="body2" color="text.secondary" sx={{ fontSize: '0.85rem' }}>
                    {info.description}
                  </Typography>
                </Box>
              </Grid>
            );
          })}
        </Grid>

        <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(251, 191, 36, 0.1)', borderRadius: 2, border: '1px solid #fbbf24' }}>
          <Typography variant="body2" color="text.secondary">
            <strong>Spot Actual:</strong> ${spot.toFixed(2)}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Los porcentajes muestran la distancia desde el precio actual. Positivo = arriba, Negativo = abajo.
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}
