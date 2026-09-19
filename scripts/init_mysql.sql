-- ---------------------------------------------------------------------------
-- 初始化 MySQL 库：仅需执行一次；业务表由 FastAPI lifespan 中 SQLAlchemy create_all 创建。
-- ---------------------------------------------------------------------------

-- 创建业务库：utf8mb4 支持 emoji 与全角标点；unicode_ci 常用排序规则
CREATE DATABASE IF NOT EXISTS xiaoyi_rag DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- （以下为可选账号示例，默认使用小皮 root 时可跳过）
-- CREATE USER IF NOT EXISTS 'xiaoyi'@'%' IDENTIFIED BY 'your_password';
-- GRANT ALL ON xiaoyi_rag.* TO 'xiaoyi'@'%';
-- FLUSH PRIVILEGES;

-- Docker 内容器访问 Windows 宿主机 MySQL 时，来源 IP 常为 172.17.0.1 / 172.18.0.1；
-- 若连接被拒，检查 MySQL 用户 Host 权限或防火墙。

-- faq_tab / legal_tab / users_tab / his_chat_tab 等 DDL 与 ORM 同步维护，
-- 不在此脚本手写 CREATE TABLE，避免与 modules/database/models.py 分叉。
