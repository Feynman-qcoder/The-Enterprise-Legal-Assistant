"""核心横切能力：集中配置与进程级环境清理。"""

from modules.core.config import ASSISTANT_NAME, Settings, get_settings  # 对外暴露配置类型与单例

__all__ = ["ASSISTANT_NAME", "Settings", "get_settings"]
