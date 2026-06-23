import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained server bundle for a slim Docker image.
  output: "standalone",
  devIndicators: false,
  outputFileTracingExcludes: {
    "/*": [
      "./.vault/**/*",
      "./src/**/*",
      "./next.config.ts",
      "./plan.md",
      "./AGENTS.md",
      "./CLAUDE.md",
      "./Dockerfile",
      "./.dockerignore",
      "./README.md",
      "./package-lock.json",
      "./tsconfig.json",
      "./eslint.config.mjs",
      "./postcss.config.mjs",
      "./scripts/**/*",
    ],
  },
};

export default nextConfig;
