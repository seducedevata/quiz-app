
import React from 'react';

interface SettingItemProps {
  label: string;
  children: React.ReactNode;
  helpText?: string;
}

export const SettingItem: React.FC<SettingItemProps> = ({ label, children, helpText }) => {
  return (
    <div className="flex items-center justify-between py-md border-b border-borderColor last:border-b-0">
      <div>
        <label className="text-body text-textPrimary block">{label}</label>
        {helpText && <p className="text-caption text-textSecondary mt-xs">{helpText}</p>}
      </div>
      <div>{children}</div>
    </div>
  );
};
