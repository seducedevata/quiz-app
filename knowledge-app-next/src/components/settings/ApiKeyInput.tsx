
import React from 'react';
import { Input } from '@/components/common/Input';

interface ApiKeyInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export const ApiKeyInput: React.FC<ApiKeyInputProps> = ({ value, onChange, placeholder }) => {
  return (
    <Input
      type="password"
      placeholder={placeholder || "Enter API Key"}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
};
