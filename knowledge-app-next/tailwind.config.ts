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
        bgPrimary: 'var(--color-bg-primary)',
        bgSecondary: 'var(--color-bg-secondary)',
        textPrimary: 'var(--color-text-primary)',
        textSecondary: 'var(--color-text-secondary)',
        borderColor: 'var(--color-border-color)',
        primaryColor: 'var(--color-primary-color)',
        primaryHover: 'var(--color-primary-hover)',
        successColor: 'var(--color-success-color)',
        dangerColor: 'var(--color-danger-color)',
        warningColor: 'var(--color-warning-color)',
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