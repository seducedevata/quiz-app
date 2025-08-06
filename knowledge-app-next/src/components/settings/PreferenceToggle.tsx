
import React from 'react';

interface PreferenceToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export const PreferenceToggle: React.FC<PreferenceToggleProps> = ({ label, checked, onChange }) => {
  return (
    <label className="flex items-center cursor-pointer">
      <div className="relative">
        <input
          type="checkbox"
          className="sr-only"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <div className="block bg-gray-600 w-14 h-8 rounded-full"></div>
        <div className="dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition"></div>
      </div>
      <div className="ml-3 text-textPrimary font-medium">
        {label}
      </div>
    </label>
  );
};
