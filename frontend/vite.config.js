import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Proxy /api to the Python backend so the frontend can use same-origin requests.
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
