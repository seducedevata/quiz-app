
import React from 'react';

interface SettingsGroupProps {
  title: string;
  description?: string;
  children: React.ReactNode;
}

export const SettingsGroup: React.FC<SettingsGroupProps> = ({ title, description, children }) => {
  return (
    <div className="mb-xl p-lg bg-bgSecondary rounded-lg">
      <h2 className="text-h2 font-h2 text-textPrimary mb-md">{title}</h2>
      {description && <p className="text-body text-textSecondary mb-lg">{description}</p>}
      {children}
    </div>
  );
};
