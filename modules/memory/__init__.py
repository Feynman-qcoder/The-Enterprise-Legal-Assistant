"""
记忆子包：判断「是否询问自身聊天历史」与可选的数据库读写。

导入策略：
- `from modules.memory import is_self_history_question` —— 可选：单元测试或自定义路由；默认 RagPipeline 已不再按此分流。
- `from modules.memory.service import persist_user_turn, DEFAULT_MEMORY_CONTEXT_LINES, …` —— 依赖 SQLAlchemy 与 `users_tab`/`his_chat_tab`。

详见仓库根目录「启动与部署.md」第 7 节、「ARCHITECTURE.md」中 `modules/memory/`。
"""

from modules.memory.history_detect import is_self_history_question

__all__ = ["is_self_history_question"]
