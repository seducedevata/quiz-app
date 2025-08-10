'use client';

import { Inter } from "next/font/google";
import "./globals.css";
import { MainLayout } from "../components/layout/MainLayout";
import { ThemeProvider } from "@/hooks/useTheme";
import { MathJaxContext } from "better-react-mathjax";
import { ScreenProvider } from "../context/ScreenContext";
import { StatusProvider } from "../context/StatusContext"; // Import StatusProvider
import ErrorBoundary from "../components/common/ErrorBoundary";
import { useEffect } from "react";

// Initialize comprehensive logging system
import "../lib/initializeLogging";

const inter = Inter({ subsets: ["latin"] });

const mathJaxConfig = {
  tex: {
    inlineMath: [['$', '$'], ['\(', '\)']],
    displayMath: [['$$', '$$'], ['\[', '\]']],
  },
  options: {
    skipHtmlTags: ['noscript', 'style', 'textarea', 'pre', 'code'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider>
          <MathJaxContext config={mathJaxConfig}>
            <ScreenProvider> {/* Wrap with ScreenProvider */}
              <StatusProvider> {/* Wrap with StatusProvider */}
                <ErrorBoundary>
                  <MainLayout>{children}</MainLayout>
                </ErrorBoundary>
              </StatusProvider>
            </ScreenProvider>
          </MathJaxContext>
        </ThemeProvider>
      </body>
    </html>
  );
}
