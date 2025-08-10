/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const base = process.env.NEXT_PUBLIC_PYTHON_BRIDGE_URL || 'http://localhost:8000';
    return [
      // Proxy app-side API calls to Flask bridge
      {
        source: '/api/:path*',
        destination: `${base}/api/:path*`,
      },
      // Optional: direct bridge path for client usage
      {
        source: '/bridge/:path*',
        destination: `${base}/:path*`,
      },
      // Health passthrough
      {
        source: '/health',
        destination: `${base}/health`,
      },
    ];
  },
};

module.exports = nextConfig;
