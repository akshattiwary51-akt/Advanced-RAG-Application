import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In dev, /api is proxied to Flask so the browser sees a single origin
// (no CORS needed). Override the target with VITE_API_PROXY if Flask runs elsewhere.
const target = process.env.VITE_API_PROXY || 'http://127.0.0.1:5000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { '/api': { target, changeOrigin: true } },
  },
})
