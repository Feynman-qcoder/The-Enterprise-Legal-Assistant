import vue from "@vitejs/plugin-vue"; // Vue 单文件组件支持
import { defineConfig } from "vite";

// 开发服务器配置：仅作用于 npm run dev，不影响生产构建产物
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000", // FastAPI 默认监听地址
        changeOrigin: true,
      },
    },
  },
});
