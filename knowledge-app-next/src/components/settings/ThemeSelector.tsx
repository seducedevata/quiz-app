
import React from 'react';
import { useTheme } from '@/hooks/useTheme';
import { Button } from '@/components/common/Button';
import { Icon } from '@/components/common/Icon';

export const ThemeSelector: React.FC = () => {
  const { isDark, toggleTheme } = useTheme();

  return (
    <Button onClick={toggleTheme} variant="secondary">
      <Icon name={isDark ? 'FaSun' : 'FaMoon'} className="mr-sm" />
      {isDark ? 'Light Mode' : 'Dark Mode'}
    </Button>
  );
};
