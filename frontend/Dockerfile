# =============================================================================
# Frontend 镜像（Task 21 Part 4）：vite 构建 → nginx 托管静态产物 + /api 反代。
# -----------------------------------------------------------------------------
# 构建两阶段：node 构建 dist → nginx(alpine) 托管；
# /api 反代到 compose 网络内的 backend:8000（替代 vite dev 代理的生产等价物）。
# =============================================================================

FROM node:20-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
# 国内镜像源加速；锁文件优先 ci，失败回落 install
RUN npm ci --registry=https://registry.npmmirror.com || npm install --registry=https://registry.npmmirror.com

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
