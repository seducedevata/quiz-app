
import React from 'react';

interface PerformanceChartProps {
  data: { name: string; value: number }[];
}

export const PerformanceChart: React.FC<PerformanceChartProps> = ({ data }) => {
  return (
    <div className="w-full h-48 bg-bgSecondary rounded-lg flex items-center justify-center text-textSecondary">
      {/* Placeholder for a chart library like Recharts or Chart.js */}
      <p>Performance Chart Placeholder</p>
    </div>
  );
};
