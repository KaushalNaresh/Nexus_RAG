/** @type {import('next').NextConfig} */
const nextConfig = {
  // Expose backend URL to the browser bundle.
  // Set NEXT_PUBLIC_API_URL in Vercel dashboard or .env.local for local dev.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
  images: { unoptimized: true },
};

export default nextConfig;
