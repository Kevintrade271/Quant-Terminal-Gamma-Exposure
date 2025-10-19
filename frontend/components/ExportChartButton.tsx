'use client';

import { Button } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import { useCallback } from 'react';
import html2canvas from 'html2canvas';

interface ExportChartButtonProps {
  chartRef?: React.RefObject<any>;
  elementRef?: React.RefObject<HTMLElement>;
  filename?: string;
}

export default function ExportChartButton({ chartRef, elementRef, filename = 'chart' }: ExportChartButtonProps) {
  const handleExport = useCallback(async () => {
    try {
      if (chartRef?.current) {
        const chart = chartRef.current as any;
        const canvas = chart.canvas as HTMLCanvasElement;

        if (!canvas) {
          console.error('Canvas not found');
          return;
        }

        const url = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `${filename}_${new Date().toISOString().split('T')[0]}.png`;
        link.href = url;
        link.click();
      } else if (elementRef?.current) {
        const canvas = await html2canvas(elementRef.current, {
          backgroundColor: '#1f2937',
          scale: 2,
        });

        const url = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `${filename}_${new Date().toISOString().split('T')[0]}.png`;
        link.href = url;
        link.click();
      } else {
        console.error('No reference available');
      }
    } catch (error) {
      console.error('Error exporting chart:', error);
    }
  }, [chartRef, elementRef, filename]);

  return (
    <Button
      variant="outlined"
      size="small"
      startIcon={<DownloadIcon />}
      onClick={handleExport}
      sx={{
        color: 'primary.main',
        borderColor: 'primary.main',
        '&:hover': {
          borderColor: 'primary.light',
          backgroundColor: 'rgba(96, 165, 250, 0.1)',
        },
      }}
    >
      Exportar PNG
    </Button>
  );
}
