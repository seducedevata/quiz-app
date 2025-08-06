
import React from 'react';

interface NavigationGroupProps {
  title: string;
  children: React.ReactNode;
}

export const NavigationGroup: React.FC<NavigationGroupProps> = ({ title, children }) => {
  return (
    <div className="mb-lg">
      <h3 className="text-h4 font-h4 text-textSecondary uppercase mb-md">{title}</h3>
      {children}
    </div>
  );
};
