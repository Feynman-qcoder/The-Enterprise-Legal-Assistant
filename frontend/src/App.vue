<!--
=============================================================================

-----------------------------------------------------------------------------
输入：登录/注册卡片输入的用户名密码；localStorage 中键 xiaoyi_user_external_id（登录成功后由后端返回值写入）。
输出：authView 视图切换（'auth' 登录卡片 ↔ 'chat' 聊天页）；聊天视图行为与方案 A 完全一致。
被谁调用：浏览器加载前端入口后由 Vite 挂载本组件；不经过 Python，仅通过 HTTP 与 FastAPI 通信。
与后端契约：POST /api/auth/{register,login} {username,password[,bind_external_id]} → {user_external_id,username}；
            聊天/历史契约不变（user_external_id 仍是唯一身份锚，方案 B 裁决 D2，下游链路零改动）。
游客语义（用户追加要求）：会话内不发送 user_external_id → 后端零落库零建档；刷新即回登录卡片。
=============================================================================
根组件（方案 B）：无身份默认展示登录/注册卡片；登录/注册/游客进入后进入聊天页；聊天页头部可登出。
-->
<script setup>
// Vue 3 编译宏：script setup 顶层变量/函数自动暴露给模板，无需 export default
import { nextTick, onMounted, ref } from "vue"; // nextTick：DOM 更新后再滚动；onMounted：挂载后加载历史；ref：响应式包装基本类型/对象

const input = ref(""); // 输入框绑定值；初始空字符串
const WELCOME_TEXT =
  "你好，我是法律助手小意。我可以结合民法典、企业知识库等为你解答。试试问我「签合同注意事项有哪些？」";
const messages = ref([
  {
    role: "assistant",
    text: WELCOME_TEXT,
  },
]); // 初始一条欢迎语；后续 push 用户/助手消息；登出时重置回这条
const loading = ref(false); // true 时禁用发送按钮，防连点
const listRef = ref(null); // 绑定模板里消息列表容器的 DOM 引用，用于 scrollTop

// 与后端 ChatRequest.user_external_id 一致：登录/注册成功写入该键；游客模式沿用 UUID 生成现状
const USER_KEY = "xiaoyi_user_external_id";

// ===== 方案 B：认证视图状态（登录 / 注册 / 游客 / 登出）=====
const authView = ref(localStorage.getItem(USER_KEY) ? "chat" : "auth"); // setup 期判定初值，避免首帧闪错视图
const authMode = ref("login"); // 'login' | 'register'：卡片内两种模式
const authForm = ref({ username: "", password: "" }); // 表单绑定
const authError = ref(""); // 401/409/429 文案或前端校验错误
const authBusy = ref(false); // 提交中防双击
const isGuest = ref(false); // 游客会话标记：true 时不发送 user_external_id → 后端不落任何消息记录（用户要求：游客消息不保存）
// 登出时暂存旧身份（仅内存变量，不落 localStorage）：随后注册带 bind_external_id 无缝升级保历史（裁决 D4）
// 不落盘原因：登出语义要求移除持久身份；重开浏览器后注册即全新账号（绑定机会仅保留在当前标签页会话内）
let lastLoggedOutExternalId = null;

// 与后端 RegisterRequest 同一正则（裁决 D7 前后端双校验）
const USERNAME_RE = /^[a-zA-Z0-9_\-]{3,32}$/;
function getOrCreateUserExternalId() {
  let id = localStorage.getItem(USER_KEY); // 尝试读取已有 id
  if (!id) {
    id = crypto.randomUUID(); // 标准 Web API 生成 UUID v4
    localStorage.setItem(USER_KEY, id); // 持久化到浏览器本地存储
  }
  return id; // 每次请求带上，后端据此关联 users_tab
}

async function scrollToBottom() {
  await nextTick(); // 等待 Vue 把新消息渲染进 DOM
  const el = listRef.value; // 取 div.messages 元素
  if (el) el.scrollTop = el.scrollHeight; // 滚动条置底，显示最新消息
}

// ===== 素材问答（OpenSpec add-attachment-query，任务 4.1-4.4）=====
// 与后端 attachment_max_bytes(默认 10MB) 对齐：前端先拦一道省一次往返；后端 413 仍是权威
const ATTACHMENT_ACCEPT = "image/jpeg,image/png,image/webp,.pdf,.docx,.txt,.md";
const ATTACHMENT_MAX_MB = 10;
const attachment = ref(null); // { file, name, size, isImage, previewUrl }；null 表示未选素材
const fileInputRef = ref(null); // 隐藏的 <input type="file"> 引用
const attachNotice = ref(""); // 选择素材时的一次性提示（超大/未选），不进消息流

function pickAttachment() {
  attachNotice.value = "";
  fileInputRef.value && fileInputRef.value.click();
}

function onFileChosen(e) {
  const f = e.target.files && e.target.files[0];
  e.target.value = ""; // 清空选择，允许再次选同一个文件
  if (!f) return;
  if (f.size > ATTACHMENT_MAX_MB * 1024 * 1024) {
    attachNotice.value = `文件过大：素材不得超过 ${ATTACHMENT_MAX_MB} MB`;
    return;
  }
  const isImage = /^image\/(jpeg|png|webp)$/.test(f.type);
  attachment.value = {
    file: f,
    name: f.name,
    size: f.size,
    isImage,
    previewUrl: isImage ? URL.createObjectURL(f) : null, // 图片做缩略预览；文档只显示文件名
  };
}

function clearAttachment() {
  if (attachment.value && attachment.value.previewUrl) {
    URL.revokeObjectURL(attachment.value.previewUrl); // 释放 blob URL，防内存泄漏
  }
  attachment.value = null;
}

function send() {
  if (loading.value) return; // 加载中一律不发（含素材路径）
  if (attachment.value) return sendWithAttachment(input.value.trim()); // 有素材：走素材端点，问题允许留空
  const q = input.value.trim(); // 无素材：原逻辑不变（空问题不发）
  if (!q) return;
  return sendPlain(q);
}

// ===== 原纯文字链路（方案 A/B 既有行为，与改造前一致）=====
async function sendPlain(q) {
  messages.value.push({ role: "user", text: q }); // 用户气泡立即出现
  input.value = ""; // 清空输入框
  loading.value = true; // 进入加载态
  messages.value.push({ role: "assistant", text: "", streaming: true }); // 先占位一条空助手消息，streaming 用于显示「思考中」
  await scrollToBottom();

  const idx = messages.value.length - 1; // 刚追加的助手消息下标
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: q, user_external_id: isGuest.value ? null : getOrCreateUserExternalId() }),
      // 游客：null → 后端 persist_user_turn 直接 no-op（不落库、不建用户行、不走记忆）；登录用户：正常身份
    }); // Vite dev 会把 /api 代理到后端，见 vite.config.js
    if (!res.ok || !res.body) {
      // HTTP 4xx/5xx 或浏览器不支持 body 流
      messages.value[idx].text = `请求失败：${res.status}`;
      messages.value[idx].error = true;
      messages.value[idx].streaming = false;
      loading.value = false;
      return;
    }
    await consumeSse(res, idx); // SSE 消费骨架抽公共函数（素材路径复用同一解析）
  } catch (e) {
    messages.value[idx].text = `网络错误：${e}`; // fetch 自身失败、断网等
    messages.value[idx].error = true;
  } finally {
    messages.value[idx].streaming = false; // 关闭「思考中」态
    loading.value = false; // 恢复按钮
    await scrollToBottom();
  }
}

// ===== 素材发送链路（任务 4.1：multipart 到 /api/chat/stream-with-attachment）=====
async function sendWithAttachment(q) {
  const att = attachment.value;
  loading.value = true;
  // 用户气泡：文件名 + 问题（留空则标注将用缺省分析）
  messages.value.push({
    role: "user",
    text: att.name + (q ? `\n${q}` : "\n（未填写问题，将使用缺省分析指令）"),
  });
  input.value = "";
  clearAttachment(); // 选中的素材已随气泡固化展示，composer 里的芯片即时清除
  messages.value.push({ role: "assistant", text: "", streaming: true, attachmentCard: null }); // attachmentCard：transcription 事件填充（任务 4.2/4.3）
  await scrollToBottom();

  const idx = messages.value.length - 1;
  try {
    const fd = new FormData();
    fd.append("file", att.file); // 字段名与后端端点签名一致
    fd.append("question", q || "");
    const res = await fetch("/api/chat/stream-with-attachment", { method: "POST", body: fd });
    if (!res.ok || !res.body) {
      // 413 超限 / 415 格式 / 503 未启用：FastAPI HTTPException 的 detail 是人话文案
      let detail = `请求失败：${res.status}`;
      try {
        const data = await res.json();
        if (data && data.detail) detail = typeof data.detail === "string" ? data.detail : detail;
      } catch { /* 非 JSON 响应体：保持状态码文案 */ }
      messages.value[idx].text = detail;
      messages.value[idx].error = true;
      messages.value[idx].streaming = false;
      loading.value = false;
      return;
    }
    await consumeSse(res, idx);
  } catch (e) {
    messages.value[idx].text = `网络错误：${e}`;
    messages.value[idx].error = true;
  } finally {
    messages.value[idx].streaming = false;
    loading.value = false;
    await scrollToBottom();
  }
}

// ===== SSE 消费骨架（原 send 内联逻辑抽出；素材路径多处理 transcription 事件，任务 4.2/4.3/4.4）=====
async function consumeSse(res, idx) {
  const reader = res.body.getReader(); // 取得 ReadableStreamDefaultReader
  const decoder = new TextDecoder(); // UTF-8 字节解码为字符串
  let buf = ""; // 累积半行：SSE 可能把一行拆成多次 read
  while (true) {
    const { done, value } = await reader.read(); // 读下一块 Uint8Array
    if (done) break; // 流结束
    buf += decoder.decode(value, { stream: true }); // stream:true 表示后续还有字节，避免多字节字符截断
    const lines = buf.split("\n"); // 按换行切
    buf = lines.pop() || ""; // 最后一段可能不完整，留到下次与后续字节拼接
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue; // 忽略 SSE 里非 data 行（如空行）
      const payload = line.slice(6).trim(); // 去掉前缀 "data: "
      if (payload === "[DONE]") continue; // 结束标记，无需 JSON 解析
      try {
        const obj = JSON.parse(payload); // 后端 json.dumps 的对象
        if (obj.type === "transcription") {
          // 素材提取结果先到（回答之前）：渲染可折叠卡片（类型/规模/截断提示/检索 query/缺省问题标注）
          messages.value[idx].attachmentCard = {
            ...obj,
            open: false, // 默认折叠，避免长预览刷屏；点击展开
          };
          await scrollToBottom();
        }
        if (obj.type === "citation_report") {
          // 引用核查报告（观察模式）：回答末尾渲染核查标记（add-citation-check 任务 6.1）
          messages.value[idx].citationBadge = obj.summary || {};
          await scrollToBottom();
        }
        if (obj.chunk) {
          messages.value[idx].text += obj.chunk; // 拼接到助手气泡
          await scrollToBottom(); // 每块更新后跟随滚动
        }
        if (obj.error) {
          // 任务 4.4：失败提示按后端文案展示（提取失败文案含可操作建议），红色样式区分
          const m = messages.value[idx];
          m.error = true;
          if (!m.text) {
            m.text = obj.error; // 尚无任何回答（提取阶段失败）：文案即正文
          } else {
            m.text += `\n[错误] ${obj.error}`; // 生成阶段失败：追加错误行
          }
          await scrollToBottom();
        }
      } catch {
        /* JSON 被 TCP 截断时不完整，忽略本次等下一行 */
      }
    }
  }
}

function onKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault(); // 阻止 textarea 默认插入换行
    send(); // 改为发送消息
  }
}

// ===== 引用核查标记（add-citation-check 任务 6.1）：按汇总着色 =====
function citationClass(summary) {
  if (!summary || !summary.total) return "citation-none";
  if (summary.grounded === summary.total) return "citation-ok";      // 全有据：绿
  if (summary.article_missing) return "citation-danger";             // 条号缺失（最高风险）：红
  return "citation-warn";                                            // 文本不符/证据外：琥珀
}

// ===== 历史回显（方案 A 最小实现，插入任务）=====
// 启动时加载当前用户最近对话：刷新/重开页面后对话延续可见。
// 语义：成功且非空 → 历史即开场（不追加欢迎语）；失败或空 → 静默保持欢迎语（可用性优先）。
async function loadHistory() {
  const id = localStorage.getItem(USER_KEY); // 已有身份才加载；新用户保持欢迎语现状
  if (!id) return;
  try {
    const res = await fetch(`/api/chat/history?user_external_id=${encodeURIComponent(id)}&limit=50`);
    if (!res.ok) {
      console.warn("历史加载失败：HTTP", res.status); // 静默降级：保持欢迎语
      return;
    }
    const data = await res.json();
    if (Array.isArray(data.items) && data.items.length > 0) {
      messages.value = data.items.flatMap((item) => [
        { role: "user", text: item.question },
        { role: "assistant", text: item.answer },
      ]); // 历史即开场：尾部不追加欢迎语
      await scrollToBottom();
    }
    // 空 items：保持欢迎语（现状不变）
  } catch (e) {
    console.warn("历史加载失败：", e); // 网络错误等：不阻塞页面，保持欢迎语
  }
}

// ===== 方案 B：认证交互（登录 / 注册 / 游客 / 登出）=====
function extractAuthError(data, status) {
  // FastAPI HTTPException → {detail: "文案"}；422 校验 → {detail: [{msg}, ...]}；兜底给状态码
  const d = data && data.detail;
  if (typeof d === "string" && d) return d;
  if (Array.isArray(d) && d.length && typeof d[0].msg === "string") return d[0].msg;
  return `请求失败：${status}`;
}

async function submitAuth() {
  authError.value = "";
  const username = authForm.value.username.trim();
  const password = authForm.value.password;
  if (!USERNAME_RE.test(username)) {
    authError.value = "用户名需 3-32 位字母、数字、下划线或连字符";
    return;
  }
  if (password.length < 8 || password.length > 72) {
    authError.value = "密码长度需 8-72 个字符";
    return;
  }
  authBusy.value = true;
  try {
    const isRegister = authMode.value === "register";
    const body = { username, password };
    if (isRegister) {
      // 绑定升级（裁决 D4）：优先读 localStorage 旧 UUID（游客会话）；登出场景读内存暂存（老匿名用户升级路径）
      const oldId = localStorage.getItem(USER_KEY) || lastLoggedOutExternalId;
      if (oldId) body.bind_external_id = oldId;
    }
    const res = await fetch(isRegister ? "/api/auth/register" : "/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      authError.value = extractAuthError(data, res.status); // 401 统一文案 / 409 / 429 直接展示，不跳转
      return;
    }
    localStorage.setItem(USER_KEY, data.user_external_id); // 裁决 D2：身份锚点写回，下游链路零改动
    lastLoggedOutExternalId = null; // 登录/绑定成功：暂存使命完成
    isGuest.value = false; // 身份确立：退出游客语义（消息开始落库）
    authForm.value = { username: "", password: "" }; // 清空表单，防后退键回显密码
    authView.value = "chat";
    await loadHistory(); // 登录即回显该账号历史（含绑定升级带入的旧历史）
  } catch (e) {
    authError.value = `网络错误：${e}`;
  } finally {
    authBusy.value = false;
  }
}

function enterAsGuest() {
  // 游客会话（用户要求）：不生成/不落 localStorage 身份 → 发消息不带 user_external_id → 后端零落库；
  // 会话仅存于内存，刷新即回到登录卡片（无持久身份可恢复）
  isGuest.value = true;
  authView.value = "chat"; // 保持欢迎语开场（无历史可加载）
}

function logout() {
  lastLoggedOutExternalId = localStorage.getItem(USER_KEY); // 暂存供本会话内注册绑定（见 submitAuth）
  localStorage.removeItem(USER_KEY); // 登出语义：移除持久身份，重开浏览器回到登录卡片
  isGuest.value = false; // 游客语义随会话一并清除
  messages.value = [{ role: "assistant", text: WELCOME_TEXT }]; // 重置为欢迎语
  input.value = "";
  authError.value = "";
  authMode.value = "login";
  authView.value = "auth";
}

onMounted(() => {
  if (authView.value === "chat") {
    loadHistory(); // 已有身份（登录过/老匿名 UUID）：方案 A 异步加载历史，不阻塞输入框
  }
  // 无身份（新访客/游客刷新后）：保持 authView='auth'（setup 期已判定），默认展示登录/注册卡片
});
</script>

<template>
  <!-- 登录/注册视图（方案 B）：新访客或登出后展示 -->
  <div v-if="authView === 'auth'" class="shell auth-shell">
    <header class="hero">
      <h1>法律助手</h1>
    </header>

    <div class="auth-card">
      <div class="auth-tabs">
        <button
          type="button"
          :class="{ active: authMode === 'login' }"
          @click="authMode = 'login'; authError = ''"
        >
          登录
        </button>
        <button
          type="button"
          :class="{ active: authMode === 'register' }"
          @click="authMode = 'register'; authError = ''"
        >
          注册
        </button>
      </div>

      <form class="auth-form" @submit.prevent="submitAuth">
        <input
          v-model="authForm.username"
          class="auth-input"
          placeholder="用户名（3-32 位字母/数字/_/-）"
          autocomplete="username"
        />
        <input
          v-model="authForm.password"
          class="auth-input"
          type="password"
          :placeholder="authMode === 'register' ? '密码（8-72 字符）' : '密码'"
          autocomplete="current-password"
        />
        <p v-if="authError" class="auth-error">{{ authError }}</p>
        <button type="submit" :disabled="authBusy">
          {{ authBusy ? "提交中…" : authMode === "login" ? "登录" : "注册" }}
        </button>
      </form>

      <button type="button" class="auth-guest" @click="enterAsGuest">游客进入</button>
    </div>
  </div>

  <!-- 聊天视图（原结构 + 头部登出按钮） -->
  <div v-else class="shell">
    <header class="hero">
      <h1>法律助手</h1>
      <button type="button" class="logout-btn" @click="logout">登出</button>
    </header>

    <main class="panel">
      <!-- 消息区：max-height 限制 + 内部滚动 -->
      <div ref="listRef" class="messages">
        <div
          v-for="(m, i) in messages"
          :key="i"
          class="row"
          :class="m.role"
        >
          <div class="avatar">{{ m.role === "user" ? "我" : "意" }}</div>
          <div class="bubble">
            <span v-if="m.streaming && !m.text && !m.attachmentCard" class="typing">法律助手小意正在思考…</span>

            <!-- 任务 4.2/4.3：transcription 可折叠卡片（回答文本之前） -->
            <div v-if="m.attachmentCard" class="attachment-card">
              <button type="button" class="attachment-toggle" @click="m.attachmentCard.open = !m.attachmentCard.open">
                <span class="attachment-kind">{{ m.attachmentCard.kind === 'image' ? '图片素材' : '文档素材' }}</span>
                <span class="attachment-chars">{{ m.attachmentCard.chars }} 字符</span>
                <span v-if="m.attachmentCard.truncated" class="attachment-warn">文档较长，仅使用前部分内容</span>
                <span class="attachment-chevron">{{ m.attachmentCard.open ? '收起' : '展开预览' }}</span>
              </button>
              <!-- 任务 4.3：缺省问题标注——不填问题上传时让用户看到系统实际使用的问题 -->
              <div v-if="m.attachmentCard.used_default_question" class="attachment-default-q">
                未填写问题，已使用缺省分析指令：「{{ m.attachmentCard.effective_question }}」
              </div>
              <div class="attachment-retrieval">
                检索依据：<span class="attachment-query">{{ m.attachmentCard.retrieval_query }}</span>
              </div>
              <pre v-if="m.attachmentCard.open" class="attachment-preview">{{ m.attachmentCard.text_preview }}</pre>
            </div>

            <div class="text" :class="{ 'text-error': m.error }">{{ m.text }}</div>

            <!-- 引用核查标记（任务 6.1 + 文案友好化 2026-09-19）：回答末尾；按状态着色与分态文案 -->
            <div v-if="m.citationBadge" class="citation-badge" :class="citationClass(m.citationBadge)">
              <template v-if="m.citationBadge.total === 0">引用核查：未检出法条引用</template>
              <template v-else-if="m.citationBadge.grounded === m.citationBadge.total">
                ✓ 引用核查：{{ m.citationBadge.total }} 处引用全部与原文相符
              </template>
              <template v-else-if="m.citationBadge.article_missing">
                ⚠ 引用警示：有 {{ m.citationBadge.article_missing }} 处引用未在检索资料中找到，请人工核实后再采信
              </template>
              <template v-else-if="m.citationBadge.law_not_in_evidence">
                ⚠ 引用提示：有 {{ m.citationBadge.law_not_in_evidence }} 处引用的来源未在本次检索资料中出现，建议核对原文
              </template>
              <template v-else>
                ⚠ 引用提示：此处为概括表述（非逐字引用原文），建议核对原文
              </template>
            </div>
          </div>
        </div>
      </div>

      <div class="composer">
        <!-- 任务 4.1：素材选择入口（图片与文档，accept 与后端白名单一致） -->
        <button
          type="button"
          class="attach-btn"
          :disabled="loading"
          title="上传图片/文档（jpg/png/webp/pdf/docx/txt/md，≤10MB）"
          @click="pickAttachment"
        >＋上传</button>
        <textarea
          v-model="input"
          rows="2"
          :placeholder="attachment ? '可针对素材提问，也可留空直接发送（Enter 发送，Shift+Enter 换行）' : '输入你的问题，Enter 发送，Shift+Enter 换行'"
          @keydown="onKey"
        />
        <!-- 有素材时允许空问题发送（任务 4.1：问题输入框允许留空） -->
        <button type="button" :disabled="loading || (!input.trim() && !attachment)" @click="send">
          {{ loading ? "生成中…" : "发送" }}
        </button>
      </div>

      <!-- 已选素材芯片：文件名/缩略预览/大小/移除（任务 4.1） -->
      <div v-if="attachment" class="attachment-chip">
        <img v-if="attachment.previewUrl" :src="attachment.previewUrl" class="chip-thumb" alt="素材预览" />
        <span class="chip-icon">{{ attachment.isImage ? '图' : '文' }}</span>
        <span class="chip-name">{{ attachment.name }}</span>
        <span class="chip-size">{{ (attachment.size / 1024).toFixed(0) }} KB</span>
        <button type="button" class="chip-remove" title="移除素材" @click="clearAttachment">✕</button>
      </div>
      <p v-if="attachNotice" class="attach-notice">{{ attachNotice }}</p>

      <!-- 隐藏的文件选择器：由「＋素材」按钮触发 -->
      <input ref="fileInputRef" type="file" :accept="ATTACHMENT_ACCEPT" hidden @change="onFileChosen" />
    </main>
  </div>
</template>

<style scoped>
/* 页面外壳：居中窄栏 + 纵向 flex */
.shell {
  max-width: 880px;
  margin: 0 auto;
  padding: 2.5rem 1.25rem 4rem;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* ===== 聊天视图白底（视觉追加，与登录视图同风格）=====
   :not(.auth-shell) 只命中聊天根节点（认证视图根为 shell auth-shell）。
   突破 880px 让白底铺满全屏；hero/panel 各自收窄回 880px 居中，内容列布局与原版一致。 */
.shell:not(.auth-shell) {
  max-width: none; /* 突破限宽：白底全屏，两侧不再露 body 暗色渐变 */
  background: #f8fafa; /* 简约白底（与 .auth-shell 同款） */
  color-scheme: light; /* 滚动条/原生控件亮色渲染 */
  color: #0f172a; /* 视图内文字基色改深 */
}

.shell:not(.auth-shell) .hero,
.shell:not(.auth-shell) .panel {
  width: 100%;
  max-width: 880px; /* 内容列仍居中 880px（承接原 .shell 限宽语义） */
  margin: 0 auto;
}

.hero {
  text-align: center;
  position: relative; /* 方案 B：登出按钮绝对定位的锚点 */
}

h1 {
  margin: 0;
  font-size: 2.25rem;
  font-weight: 600;
  background: linear-gradient(120deg, #0f766e, #4338ca); /* 深色渐变字：白底上保持对比 */
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #ffffff; /* 白卡片（与登录卡片同风格） */
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 1rem;
  box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06); /* 柔和浅阴影（白底无需玻璃拟态） */
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem;
  max-height: min(58vh, 640px);
  scroll-behavior: smooth;
}

.row {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1rem;
  align-items: flex-start;
}

.row.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  font-size: 0.85rem;
  font-weight: 600;
  flex-shrink: 0;
  background: #f1f5f9; /* 浅灰底 */
  border: 1px solid #e2e8f0;
}

.row.user .avatar {
  background: var(--bubble-user);
  border-color: rgba(94, 234, 212, 0.35);
}

.row.assistant .avatar {
  background: var(--bubble-bot);
  border-color: rgba(129, 140, 248, 0.4);
}

.bubble {
  max-width: 78%;
  padding: 0.85rem 1rem;
  border-radius: var(--radius);
  line-height: 1.65;
  font-size: 0.95rem;
  border: 1px solid #e2e8f0;
}

.row.user .bubble {
  background: var(--bubble-user);
  border-color: rgba(94, 234, 212, 0.25);
}

.row.assistant .bubble {
  background: #f1f5f9; /* 助手气泡浅灰（白底简约风；用户气泡保留浅青着色） */
}

.text {
  white-space: pre-wrap;
  word-break: break-word;
}

.typing {
  color: #64748b; /* 白底可读的 slate 灰 */
  font-size: 0.9rem;
}

.composer {
  display: grid;
  grid-template-columns: auto 1fr auto; /* 任务 4.1：三列——素材按钮 / 输入框 / 发送 */
  gap: 0.75rem;
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid #e2e8f0;
}

textarea {
  width: 100%;
  resize: none;
  border-radius: 14px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #0f172a;
  padding: 0.75rem 1rem;
  font: inherit;
  outline: none;
}

textarea::placeholder {
  color: #94a3b8; /* 白底输入框浅灰占位符 */
}

textarea:focus {
  border-color: rgba(94, 234, 212, 0.45);
  box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.12);
}

button {
  border: none;
  border-radius: 14px;
  padding: 0 1.35rem;
  font-weight: 600;
  cursor: pointer;
  color: #0f172a;
  background: linear-gradient(130deg, #5eead4, #818cf8);
  box-shadow: 0 10px 30px rgba(94, 234, 212, 0.2);
  transition: transform 0.12s ease, opacity 0.12s ease;
}

button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

button:not(:disabled):hover {
  transform: translateY(-1px);
}

/* ===== 方案 B：登录/注册卡片 ===== */
/* 视觉追加（纯 CSS）：全屏白底 + 双轴居中——修偏左（.shell 的 align-items:stretch 把卡片拉到上限后停左缘）
   与暗底残留（body 暗色渐变 + 暗色玻璃卡片）；同特异性下本规则晚于 .shell 声明，覆盖生效 */
.auth-shell {
  justify-content: center; /* 垂直居中（原有） */
  align-items: center; /* 水平居中：flex column 交叉轴对齐，修偏左根因 */
  max-width: none; /* 突破 .shell 的 880px 限宽，白底铺满全屏 */
  background: #f8fafa; /* 简约白底：min-height:100vh 不透明背景盖住 body 暗色渐变 */
  color-scheme: light; /* 输入框/autofill 等原生控件用亮色渲染 */
  color: #0f172a; /* 视图内文字基色改深（覆盖继承的全局亮色 --text） */
}

.auth-card {
  width: 100%;
  max-width: 380px;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  background: #ffffff; /* 白卡片 */
  border: 1px solid #e2e8f0; /* 细边框 */
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06); /* 柔和浅阴影（白底无需玻璃拟态，删 backdrop-filter） */
}

.auth-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
}

.auth-tabs button {
  padding: 0.55rem 0;
  font-size: 0.95rem;
  background: #f1f5f9; /* 未选中：浅灰底 */
  color: #475569;
  border: 1px solid #e2e8f0;
  box-shadow: none;
}

.auth-tabs button.active {
  background: linear-gradient(130deg, #5eead4, #818cf8); /* 选中：主渐变 */
  color: #0f172a;
}

.auth-form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.auth-input {
  width: 100%;
  border-radius: 14px;
  border: 1px solid #cbd5e1;
  background: #ffffff;
  color: #0f172a;
  padding: 0.75rem 1rem;
  font: inherit;
  outline: none;
}

.auth-input::placeholder {
  color: #94a3b8; /* 白底输入框浅灰占位符 */
}

.auth-input:focus {
  border-color: rgba(94, 234, 212, 0.45);
  box-shadow: 0 0 0 3px rgba(94, 234, 212, 0.12);
}

.auth-error {
  margin: 0;
  color: #dc2626; /* 白底深红（原亮红 #f87171 在白底对比不足） */
  font-size: 0.85rem;
}

.auth-guest {
  padding: 0.55rem 0;
  font-size: 0.9rem;
  background: #f9fafb;
  color: #475569;
  border: 1px solid #e2e8f0;
  box-shadow: none;
}

/* ===== 方案 B：聊天页头部登出按钮 ===== */
.logout-btn {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  padding: 0.45rem 1.1rem;
  font-size: 0.85rem;
  font-weight: 500;
  background: #f1f5f9;
  color: #475569;
  border: 1px solid #e2e8f0;
  box-shadow: none;
}

.logout-btn:not(:disabled):hover {
  color: #0f172a; /* 白底深字（原亮色 --text 在白底不可见） */
  border-color: rgba(94, 234, 212, 0.4);
}

/* ===== 素材问答（add-attachment-query，任务 4.1-4.4）===== */
/* 素材按钮：次级按钮风格（同 .auth-guest 一族），窄列 */
.attach-btn {
  padding: 0 0.9rem;
  font-size: 0.85rem;
  background: #f1f5f9;
  color: #475569;
  border: 1px solid #e2e8f0;
  box-shadow: none;
  white-space: nowrap;
}

.attach-btn:not(:disabled):hover {
  color: #0f172a;
  border-color: rgba(94, 234, 212, 0.4);
}

/* 已选素材芯片：一行展示 预览图/类型图标/文件名/大小/移除 */
.attachment-chip {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.6rem;
  padding: 0.5rem 0.75rem;
  background: #f8fafc;
  border: 1px dashed #cbd5e1;
  border-radius: 12px;
  font-size: 0.85rem;
  color: #475569;
}

.chip-thumb {
  width: 36px;
  height: 36px;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.chip-icon {
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: #eef2ff;
  color: #4338ca;
  font-size: 0.8rem;
  font-weight: 600;
  flex-shrink: 0;
}

.chip-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #0f172a;
}

.chip-size {
  color: #94a3b8;
  flex-shrink: 0;
}

.chip-remove {
  padding: 0 0.5rem;
  font-size: 0.8rem;
  background: transparent;
  color: #94a3b8;
  border: none;
  box-shadow: none;
  flex-shrink: 0;
}

.chip-remove:hover {
  color: #dc2626;
  transform: none;
}

.attach-notice {
  margin: 0.4rem 0 0;
  font-size: 0.82rem;
  color: #dc2626;
}

/* transcription 可折叠卡片（回答之前）：浅底卡片 + 顶部摘要行 */
.attachment-card {
  margin-bottom: 0.75rem;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #f8fafc;
  overflow: hidden;
}

.attachment-toggle {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  width: 100%;
  padding: 0.5rem 0.75rem;
  background: transparent;
  border: none;
  box-shadow: none;
  font-size: 0.82rem;
  color: #475569;
  text-align: left;
  border-radius: 0;
}

.attachment-toggle:hover {
  transform: none;
  background: #f1f5f9;
}

.attachment-kind {
  font-weight: 600;
  color: #4338ca;
  flex-shrink: 0;
}

.attachment-chars {
  color: #64748b;
  flex-shrink: 0;
}

.attachment-warn {
  color: #b45309; /* 琥珀色：截断提示（任务 4.2 验收文案） */
  background: #fef3c7;
  border-radius: 8px;
  padding: 0.1rem 0.5rem;
  font-size: 0.78rem;
  flex-shrink: 0;
}

.attachment-chevron {
  margin-left: auto;
  color: #94a3b8;
  flex-shrink: 0;
}

.attachment-default-q {
  padding: 0.35rem 0.75rem;
  font-size: 0.8rem;
  color: #0f766e;
  background: #ecfdf5;
  border-top: 1px solid #e2e8f0;
}

.attachment-retrieval {
  padding: 0.35rem 0.75rem 0.5rem;
  font-size: 0.8rem;
  color: #64748b;
  border-top: 1px solid #e2e8f0;
  word-break: break-all;
}

.attachment-query {
  color: #0f172a;
}

.attachment-preview {
  margin: 0;
  padding: 0.6rem 0.75rem;
  max-height: 220px;
  overflow-y: auto;
  font-size: 0.78rem;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
  color: #334155;
  background: #ffffff;
  border-top: 1px solid #e2e8f0;
}

/* 失败文案（任务 4.4）：红色系正文 */
.text-error {
  color: #dc2626;
}

/* ===== 引用核查标记（add-citation-check 任务 6.1）===== */
.citation-badge {
  margin-top: 0.5rem;
  padding: 0.3rem 0.7rem;
  border-radius: 10px;
  font-size: 0.78rem;
  line-height: 1.5;
  border: 1px solid transparent;
}

.citation-ok {
  background: #ecfdf5;
  color: #047857;
  border-color: #a7f3d0;
}

.citation-warn {
  background: #fffbeb;
  color: #b45309;
  border-color: #fde68a;
}

.citation-danger {
  background: #fef2f2;
  color: #b91c1c;
  border-color: #fecaca;
}

.citation-none {
  background: #f8fafc;
  color: #94a3b8;
  border-color: #e2e8f0;
}


</style>
