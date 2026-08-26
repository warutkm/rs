/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: '/api/v2/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/v2/:path*`,
      },
      {
        source: '/api/metrics',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/metrics`,
      },
      {
        source: '/api/admin/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/admin/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
