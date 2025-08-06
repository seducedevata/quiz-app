'use client';

import React from 'react';

interface DropdownOption {
  label: string;
  value: string;
}

interface DropdownProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: DropdownOption[];
}

export const Dropdown: React.FC<DropdownProps> = ({ label, id, options, ...props }) => {
  return (
    <div className="form-group">
      {label && <label htmlFor={id}>{label}</label>}
      <select id={id} className="form-select" {...props}>
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
};