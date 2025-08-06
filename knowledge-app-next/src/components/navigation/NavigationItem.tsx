
'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Icon } from '@/components/common/Icon';

interface NavigationItemProps {
  href: string;
  icon: string;
  text: string;
}

export const NavigationItem: React.FC<NavigationItemProps> = ({ href, icon, text }) => {
  const pathname = usePathname();
  const isActive = pathname === href;

  const activeClasses = isActive ? 'bg-primaryColor text-white' : 'text-textSecondary hover:bg-bgPrimary hover:translate-x-1';

  return (
    <Link href={href} className={`flex items-center p-md rounded-md mb-sm text-body transition-all duration-300 ease-in-out ${activeClasses}`}>
      <Icon name={icon} className="mr-md" />
      {text}
    </Link>
  );
};
