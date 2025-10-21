'use client';

import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import { useEffect, useState } from 'react';
import { fetchStatus } from '@/lib/api';
import type { StatusResponse } from '@/lib/types';

interface TickerCardProps {
  ticker: string;
  label: string;
  isSelected: boolean;
  onClick: () => void;
}

const tickerInfo: Record<string, { emoji: string; description: string; category: string }> = {
  SPY: { emoji: '📈', description: 'S&P 500 - Más líquido', category: 'Índice' },
  QQQ: { emoji: '💻', description: 'Nasdaq 100 - Tech', category: 'Índice' },
  DIA: { emoji: '🏭', description: 'Dow Jones - Industriales', category: 'Índice' },
  IWM: { emoji: '🏢', description: 'Russell 2000 - Small Caps', category: 'Índice' },
  GLD: { emoji: '🥇', description: 'Gold - Oro', category: 'Commodity' },
  SLV: { emoji: '🥈', description: 'Silver - Plata', category: 'Commodity' },
};

export default function TickerCard({ ticker, label, isSelected, onClick }: TickerCardProps) {
  const info = tickerInfo[ticker] || { emoji: '📊', description: label, category: 'ETF' };
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const data = await fetchStatus(ticker);
        setStatus(data);
      } catch (error) {
        console.error(`Error loading status for ${ticker}:`, error);
      } finally {
        setLoading(false);
      }
    };

    loadStatus();
    const interval = setInterval(loadStatus, 60000);
    return () => clearInterval(interval);
  }, [ticker]);

  const getVixColor = (zscore: number) => {
    if (zscore > 1) return '#ef4444';
    if (zscore > 0.5) return '#f59e0b';
    if (zscore < -0.5) return '#10b981';
    return '#6b7280';
  };

  return (
    <Card
      onClick={onClick}
      sx={{
        bgcolor: isSelected ? 'primary.main' : 'background.paper',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        border: isSelected ? '2px solid' : '1px solid',
        borderColor: isSelected ? 'primary.main' : '#374151',
        transform: isSelected ? 'scale(1.05)' : 'scale(1)',
        '&:hover': {
          transform: 'scale(1.05)',
          borderColor: 'primary.main',
          boxShadow: isSelected ? 4 : 3,
        },
        boxShadow: isSelected ? 3 : 1,
        minHeight: '140px',
      }}
    >
      <CardContent sx={{ p: 2, display: 'flex', flexDirection: 'column', height: '100%' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
          <Typography variant="h4" sx={{ fontSize: '2rem' }}>
            {info.emoji}
          </Typography>
          <Chip
            label={info.category}
            size="small"
            sx={{
              bgcolor: isSelected ? 'rgba(255,255,255,0.2)' : 'rgba(96, 165, 250, 0.1)',
              color: isSelected ? '#fff' : 'primary.main',
              fontWeight: 600,
              fontSize: '0.7rem',
            }}
          />
        </Box>

        <Typography
          variant="h6"
          sx={{
            color: isSelected ? '#fff' : 'text.primary',
            fontWeight: 700,
            mb: 0.5,
          }}
        >
          {ticker}
        </Typography>

        <Typography
          variant="body2"
          sx={{
            color: isSelected ? 'rgba(255,255,255,0.9)' : 'text.secondary',
            fontSize: '0.75rem',
            lineHeight: 1.3,
            mb: 1,
          }}
        >
          {info.description}
        </Typography>

        {!loading && status && (
          <Box sx={{ mt: 'auto' }}>
            <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5, mb: 0.5 }}>
              <Typography
                variant="body2"
                sx={{
                  color: isSelected ? '#fff' : 'text.primary',
                  fontWeight: 700,
                  fontSize: '0.9rem',
                }}
              >
                ${status.spot.toFixed(2)}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              {ticker === 'SPY' && (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.3,
                    bgcolor: isSelected ? 'rgba(255,255,255,0.2)' : getVixColor(status.vix_zscore) + '20',
                    px: 0.8,
                    py: 0.3,
                    borderRadius: 1,
                  }}
                >
                  <Typography
                    variant="caption"
                    sx={{
                      color: isSelected ? '#fff' : getVixColor(status.vix_zscore),
                      fontWeight: 600,
                      fontSize: '0.65rem',
                    }}
                  >
                    VIX: {status.vix_current.toFixed(1)}
                  </Typography>
                </Box>
              )}

              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.3,
                  bgcolor: isSelected ? 'rgba(255,255,255,0.2)' : 'rgba(96, 165, 250, 0.1)',
                  px: 0.8,
                  py: 0.3,
                  borderRadius: 1,
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    color: isSelected ? '#fff' : 'primary.main',
                    fontWeight: 600,
                    fontSize: '0.65rem',
                  }}
                >
                  En vivo
                </Typography>
              </Box>

              {isSelected && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.3 }}>
                  <TrendingUpIcon sx={{ fontSize: '0.85rem', color: '#fff' }} />
                  <Typography variant="caption" sx={{ color: '#fff', fontWeight: 600, fontSize: '0.65rem' }}>
                    Activo
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
        )}

        {loading && (
          <Typography
            variant="caption"
            sx={{
              color: isSelected ? 'rgba(255,255,255,0.7)' : 'text.secondary',
              fontSize: '0.65rem',
              mt: 1,
            }}
          >
            Cargando...
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}
