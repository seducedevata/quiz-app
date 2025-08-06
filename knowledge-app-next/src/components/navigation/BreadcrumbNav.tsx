
import React from 'react';
import Link from 'next/link';

interface BreadcrumbItem {
  label: string;
  href: string;
}

interface BreadcrumbNavProps {
  items: BreadcrumbItem[];
}

export const BreadcrumbNav: React.FC<BreadcrumbNavProps> = ({ items }) => {
  return (
    <nav className="text-textSecondary text-bodySmall mb-lg">
      {items.map((item, index) => (
        <span key={item.href}>
          <Link href={item.href} className="hover:underline">
            {item.label}
          </Link>
          {index < items.length - 1 && <span className="mx-xs">/</span>}
        </span>
      ))}
    </nav>
  );
};
