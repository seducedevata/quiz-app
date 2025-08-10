import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: 'class',
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Map to the actual CSS variables defined in globals.css
        'bg-primary': 'var(--bg-primary)',
        'bg-secondary': 'var(--bg-secondary)',
        'bg-tertiary': 'var(--bg-tertiary)',
        'text-primary': 'var(--text-primary)',
        'text-secondary': 'var(--text-secondary)',
        'text-muted': 'var(--text-muted)',
        'text-inverse': 'var(--text-inverse)',
        'border-color': 'var(--border-color)',
        'primary-color': 'var(--primary-color)',
        'primary-hover': 'var(--primary-hover)',
        'success-color': 'var(--success-color)',
        'danger-color': 'var(--danger-color)',
        'warning-color': 'var(--warning-color)',
        'info-color': 'var(--info-color)',
      },
      spacing: {
        xs: '4px',
        sm: '8px',
        md: '16px',
        lg: '24px',
        xl: '32px',
        xxl: '48px',
      },
      borderRadius: {
        sm: '4px',
        md: '8px',
        lg: '12px',
        xl: '16px',
      },
      fontSize: {
        h1: '28px',
        h2: '24px',
        h3: '20px',
        h4: '18px',
        body: '16px',
        bodySmall: '14px',
        caption: '12px',
        button: '16px',
      },
      fontWeight: {
        h1: '700',
        h2: '600',
        h3: '600',
        h4: '600',
        body: '400',
        bodySmall: '400',
        caption: '400',
        button: '500',
      },
    },
  },
  plugins: [],
};
export default config;