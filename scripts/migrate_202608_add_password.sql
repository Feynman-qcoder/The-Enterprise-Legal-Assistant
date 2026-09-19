-- ---------------------------------------------------------------------------
-- 迁移（方案 B 登录注册，2026-08）：users_tab 增加密码哈希列。
--
-- 执行方式：存量库手动执行一次，例如：
--   mysql -h127.0.0.1 -uxiaoyi -p xiaoyi_rag < scripts/migrate_202608_add_password.sql
--   （或任意客户端执行下方唯一一条 ALTER）
-- 全新库无需执行：FastAPI lifespan 的 create_all 会按 ORM（UserTab.password_hash）自动建列。
-- 幂等性：重复执行报 Duplicate column name 'password_hash'，属预期报错可忽略。
-- 回滚：ALTER TABLE users_tab DROP COLUMN password_hash;
--
-- 语义：NULL = 匿名用户（不可登录）；存量 49 行全部保持 NULL，零影响。
-- 哈希格式（自描述，迭代数可演进）：pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
-- ---------------------------------------------------------------------------

ALTER TABLE users_tab ADD COLUMN password_hash VARCHAR(255) NULL;

-- 验证：DESC users_tab; 应出现 password_hash / varchar(255) / YES / NULL
