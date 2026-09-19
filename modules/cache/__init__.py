"""Redis 缓存子模块：导出缓存类与客户端工厂。"""

from modules.cache.redis_client import RedisCache, get_redis  # 对外 API

__all__ = ["RedisCache", "get_redis"]
