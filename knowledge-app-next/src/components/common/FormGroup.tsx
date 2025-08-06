'use client';

import React from 'react';

interface FormGroupProps {
  label?: string;
  children: React.ReactNode;
  htmlFor?: string;
}

export const FormGroup: React.FC<FormGroupProps> = ({ label, children, htmlFor }) => {
  return (
    <div className="form-group">
      {label && <label htmlFor={htmlFor}>{label}</label>}
      {children}
    </div>
  );
};
