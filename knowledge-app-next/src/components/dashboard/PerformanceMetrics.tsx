'use client';

import React from 'react';
import { Card } from '@/components/common/Card';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PerformanceMetricsProps {
  data: {
    labels: string[];
    datasets: {
      label: string;
      data: number[];
      borderColor: string;
      backgroundColor: string;
    }[];
  };
}

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ data }) => {
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Quiz Performance Over Time',
      },
    },
  };

  return (
    <Card className="mt-lg">
      <h3 className="text-h3 font-h3 text-textPrimary mb-lg">Performance Metrics</h3>
      <div className="h-80">
        <Line options={options} data={data} />
      </div>
    </Card>
  );
};