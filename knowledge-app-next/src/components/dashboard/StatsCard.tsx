import React from 'react';
import { Card } from '@/components/common/Card';

interface StatsCardProps {
  title: string;
  subtitle: string;
}

export const StatsCard: React.FC<StatsCardProps> = ({ title, subtitle }) => {
  return (
    <Card>
      <h2 className="text-h2 font-h2 text-textPrimary m-0">{title}</h2>
      <p className="text-body text-textSecondary m-0">{subtitle}</p>
    </Card>
  );
};