'use client';

import React from 'react';

interface IconProps {
  name: string; // This could be a string representing an emoji or a class name for an icon font
  className?: string;
}

export const Icon: React.FC<IconProps> = ({ name, className }) => {
  return (
    <span className={`icon ${className || ''}`}>
      {name}
    </span>
  );
};
