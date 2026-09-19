# =============================================================================
# -----------------------------------------------------------------------------
# 输入：`ParsedDocument`（来自 parse_document）。
# 输出：清洗后的 `ParsedDocument`（text/segments 已去噪，metadata 追加 cleaning 统计）。
# 被谁调用：在线 incremental.ingest_legal_document、离线 mysql_loaders（Phase C 接入后）；
#          本模块本身无副作用，绝不修改数据库、绝不修改 Raw Source。
# 设计原则：deterministic / rule-based / 保守 —— 只删确定噪声，保留全部法律语义。
# =============================================================================
"""
Legal Corpus Cleaning & Normalization V1（确定性、可解释、可测试、可回滚）。

处理目标（保守）：
  1. 政府网页派生 MD/TXT 的导航/分享/二维码/Footer 等 Boilerplate；
  2. PDF 跨页重复页眉/页脚（基于页边界统计）；
  3. 合同模板大量下划线/横线占位符 → [待填写]；
  4. 独立页码（第 3 页 / - 12 -）但保留法条编号（第三条、3.、（三））；
  5. 多余空白/空行/无信息行；
  6. 极低信息块（纯横线/纯组件字符串）。

明确不做（V1）：
  LLM 改写/摘要/补全；简繁转换；法律数字改写；中英文标点大规模重写；
  DOC/DOCX 格式转换；OCR；Chunk 参数调整；任何数据库写入。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from modules.ingestion.document_parsing import ParsedDocument

# =============================================================================
# 配置常量（阈值集中，便于调参与审计；不写死在业务逻辑分支里）
# =============================================================================
# 超过该移除比例即标 NEEDS_REVIEW（不直接 REJECT，等待人工确认）
REMOVED_RATIO_REVIEW_THRESHOLD = 0.30
# 清洗后正文低于该字符数（且原文非空）→ NEEDS_REVIEW
MIN_CLEAN_CHARS_REVIEW = 50

# PDF 跨页页眉/页脚检测：只检查每页前/后 N 行
PDF_BOUNDARY_SCAN_LINES = 3
# 标准化后同一短文本在 >= 该比例页面边界重复出现，判定为页眉/页脚
PDF_REPEATED_BOUNDARY_RATIO = 0.60
# PDF 至少需要多少页才启用页眉/页脚检测（避免短文档误判）
PDF_MIN_PAGES_FOR_BOUNDARY = 4

CLEANING_VERSION = "v1.1"


# =============================================================================
# Boilerplate 规则（保守；基于真实 Corpus 污染样本校准）
# =============================================================================
# 整行精确匹配（strip 后完全相等）即删除
BOILERPLATE_EXACT_LINES = frozenset(
    {
        "JiaThis Button BEGIN",
        "JiaThis Button END",
        "Baidu Button BEGIN",
        "Baidu Button END",
        "二维码生成专用",
        "扫一扫在手机打开当前页",
        "扫一扫在手机上打开当前页",
        "关闭窗口",
        "打印",
        "返回顶部",
        "侧边栏飘窗",
        "footer",
        "header",
    }
)

# 整行前缀匹配即删除（网页组件标记）
BOILERPLATE_LINE_PREFIXES = (
    "start component HTML组件",
    "end component HTML组件",
    "JiaThis Button",
    "Baidu Button",
)

# 整行安全正则（锚定整行，避免误伤正文）
BOILERPLATE_SAFE_REGEX = [
    # 网页组件标记（含 header / footer / 访问量统计 等任意括注）
    re.compile(r"^\s*start component HTML组件\s*\(", re.UNICODE),
    re.compile(r"^\s*end component HTML组件", re.UNICODE),
    # 二维码 / 扫一扫
    re.compile(r"^\s*二维码生成专用", re.UNICODE),
    re.compile(r"^\s*扫一扫在手机(上)?打开当前页", re.UNICODE),
    # 分享挂件（仅明确挂件上下文，不碰「分享经济」等正文）
    re.compile(r"^\s*(分享到|一键分享|分享按钮|分享至)\b", re.UNICODE),
    # 上一篇 / 下一篇 导航
    re.compile(r"^\s*(上一篇|下一篇)\s*[:：]", re.UNICODE),
    # 责任编辑
    re.compile(r"^\s*责任编辑\s*[:：]", re.UNICODE),
    # 含 javascript: 的打印/纠错链接行（如 Baidu Button END [【打印】](javascript:...)【纠错】）
    re.compile(r"javascript\s*:", re.UNICODE),
    # 纯导航行：| 打印 |  | 关闭窗口 | 这类由 junk token 组成的表格行
    re.compile(r"^\s*\|[\s|]*(打印|关闭窗口|纠错|分享|收藏|返回顶部)[\s|]*$", re.UNICODE),
    # CMS 生成器页脚
    re.compile(r"Produced By CMS", re.UNICODE),
    re.compile(r"网站群内容管理系统", re.UNICODE),
    re.compile(r"publishdate\s*:", re.UNICODE),
    # 返回顶部 / 侧边栏飘窗 导航行
    re.compile(r"返回顶部", re.UNICODE),
    re.compile(r"侧边栏飘窗", re.UNICODE),
    # 页脚装饰导航：- 学习强国 ◆ ◆
    re.compile(r"^\s*[-*]\s*[\u4e00-\u9fffA-Za-z0-9]+\s*◆+\s*◆+\s*$", re.UNICODE),
    # 网站备案/版权页脚家具（确定非法律正文）
    re.compile(r"京ICP备", re.UNICODE),
    re.compile(r"京公网安备", re.UNICODE),
    re.compile(r"版权所有", re.UNICODE),
    re.compile(r"^\s*PC版\s*$", re.UNICODE),
    # CMS 解析器留下的组件边界标记；整行锚定，避免误伤正文中的 component 单词。
    re.compile(r"^\s*(?:start|end)\s+component(?:\s+.*)?$", re.IGNORECASE | re.UNICODE),
    # JiaThis 分享链接；只匹配独立 URL / Markdown link 行，不删除正文中的普通 URL。
    re.compile(
        r"^\s*(?:\[[^\]]*\]\()?https?://(?:www\.)?jiathis\.com(?:/[^\s)]*)?\)?\s*$",
        re.IGNORECASE | re.UNICODE,
    ),
    # 含 baidu 技术 token 的纯装饰线；要求 token 两侧均为至少三个装饰字符。
    re.compile(r"^\s*[-_=~—–─]{3,}\s*baidu\s*[-_=~—–─]{3,}\s*$", re.IGNORECASE | re.UNICODE),
]

# 网页派生的「纯 junk 单元格」集合（用于拆表后判断整行是否为导航）
JUNK_NAV_TOKENS = frozenset(
    {"打印", "关闭窗口", "纠错", "分享", "收藏", "返回顶部", "上一篇", "下一篇", ""}
)

RESIDUAL_TEXT_PATTERNS = (
    ("jiathis.com", re.compile(r"jiathis\.com", re.IGNORECASE | re.UNICODE)),
    ("JiaThis Button", re.compile(r"JiaThis\s+Button", re.IGNORECASE | re.UNICODE)),
    ("二维码生成专用", re.compile(r"二维码生成专用", re.UNICODE)),
    ("Produced By CMS", re.compile(r"Produced\s+By\s+CMS", re.IGNORECASE | re.UNICODE)),
    ("访问量统计", re.compile(r"访问量统计", re.UNICODE)),
    ("网站群内容管理系统", re.compile(r"网站群内容管理系统", re.UNICODE)),
    ("publishdate", re.compile(r"publishdate\s*:", re.IGNORECASE | re.UNICODE)),
)

# 手写/渲染分隔符（纯横线、纯点线）—— 整行仅含这些字符视为无信息
PURE_SEPARATOR_RE = re.compile(r"^[\s_＿\-—–・─=~．.•·]+$", re.UNICODE)

# 独立页码（仅整行完全匹配才删，绝不碰法条编号）
PAGE_NUMBER_REGEX = [
    re.compile(r"^\s*第\s*\d+\s*页\s*/\s*共\s*\d+\s*页\s*$", re.UNICODE),
    re.compile(r"^\s*第\s*\d+\s*页\s*$", re.UNICODE),
    re.compile(r"^\s*[—\-–]\s*\d+\s*[—\-–]\s*$", re.UNICODE),
]

# 合同占位符：字段 + 结尾下划线 → 字段：[待填写]
PLACEHOLDER_FIELD_RE = re.compile(
    r"^(?P<label>\s*[\u4e00-\u9fffA-Za-z0-9（）()【】\[\]、，,.:：·\-/]+)\s*[：:]\s*[_＿]{2,}\s*$",
    re.UNICODE,
)
# 合同占位符：无冒号短标签 + 结尾下划线 → 标签：[待填写]（如「其他________」）
PLACEHOLDER_LABEL_RE = re.compile(
    r"^(?P<label>\s*[\u4e00-\u9fffA-Za-z0-9（）()【】]+)\s*[_＿]{2,}\s*$",
    re.UNICODE,
)
# 日期占位符：____年____月____日 → 日期：[待填写]
PLACEHOLDER_DATE_RE = re.compile(r"^\s*[_＿\s]*年[_＿\s]*月[_＿\s]*日[_＿\s]*$", re.UNICODE)


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class CleaningStats:
    """每个文档的清洗统计，序列化进 metadata['cleaning'] 与 manifest。"""

    raw_chars: int = 0
    clean_chars: int = 0
    removed_chars: int = 0
    removed_ratio: float = 0.0
    boilerplate_hits: int = 0
    placeholder_normalized: int = 0
    blank_lines_removed: int = 0
    low_info_blocks_removed: int = 0
    normalized_sha256: str = ""
    quality_status: str = "PASS"  # PASS | NEEDS_REVIEW
    quality_reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.quality_reasons is None:
            self.quality_reasons = []


# =============================================================================
# 行级清洗原语
# =============================================================================
def _is_boilerplate_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s in BOILERPLATE_EXACT_LINES:
        return True
    if any(s.startswith(p) for p in BOILERPLATE_LINE_PREFIXES):
        return True
    if _is_markdown_junk_row(s):
        return True
    if any(rx.search(s) for rx in BOILERPLATE_SAFE_REGEX):
        return True
    return False


def _is_markdown_junk_row(line: str) -> bool:
    """Return True only when every non-empty Markdown cell is a known web action token."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [cell.strip() for cell in stripped[1:-1].split("|")]
    nonempty = [cell for cell in cells if cell]
    return bool(nonempty) and all(cell in JUNK_NAV_TOKENS for cell in nonempty)


def _is_markdown_separator_row(line: str) -> bool:
    """Recognize a Markdown alignment row without treating ordinary tables as junk by itself."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [cell.strip() for cell in stripped[1:-1].split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell) and all(cells)


def _find_residual_boilerplate(text: str) -> tuple[int, list[str]]:
    """Scan final clean text for high-confidence residual web/CMS noise."""
    hits = 0
    patterns: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\s*(?:start|end)\s+component(?:\s+.*)?$", stripped, re.IGNORECASE):
            hits += 1
            patterns.add("component_marker")
        if _is_markdown_junk_row(stripped):
            hits += 1
            patterns.add("markdown_junk_row")
        if stripped in {"打印", "关闭窗口"}:
            hits += 1
            patterns.add(f"standalone_{stripped}")
        if re.match(r"^\s*[-_=~—–─]{3,}\s*baidu\s*[-_=~—–─]{3,}\s*$", stripped, re.IGNORECASE):
            hits += 1
            patterns.add("baidu_decorative_line")
        for name, regex in RESIDUAL_TEXT_PATTERNS:
            if regex.search(stripped):
                hits += 1
                patterns.add(name)
    return hits, sorted(patterns)


def _is_pure_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and len(s) >= 3 and bool(PURE_SEPARATOR_RE.match(s))


def _is_standalone_page_number(line: str) -> bool:
    s = line.strip()
    return any(rx.match(s) for rx in PAGE_NUMBER_REGEX)


def _normalize_placeholder_in_line(line: str) -> tuple[str, int]:
    """对单行做合同占位符标准化；返回 (新行, 命中数)。"""
    hits = 0
    s = line.strip()
    # 日期占位符优先
    if PLACEHOLDER_DATE_RE.match(s):
        return "日期：[待填写]", 1
    # 字段 + 冒号 + 下划线
    m = PLACEHOLDER_FIELD_RE.match(line)
    if m:
        label = m.group("label").strip()
        return f"{label}：[待填写]", 1
    # 短标签 + 下划线（无冒号）
    m = PLACEHOLDER_LABEL_RE.match(line)
    if m:
        label = m.group("label").strip()
        if label:  # 必须有语义标签，避免把纯横线误改成 [待填写]
            return f"{label}：[待填写]", 1
    return line, hits


def _clean_line(line: str, *, count: dict[str, int], normalize_placeholder: bool) -> str | None:
    """
    处理单行，返回清洗后行；若整行应删除返回 None。
    count: 累计统计字典（boilerplate_hits / blank_lines_removed / low_info_blocks_removed /
                          placeholder_normalized）。
    """
    stripped = line.strip()
    # 1) 空行统计（在 boilerplate 判断前，避免把空行计入 boilerplate）
    if not stripped:
        return line  # 空白压缩在段落级处理；此处保留交由 normalize_ws 合并
    # 2) Boilerplate
    if _is_boilerplate_line(line):
        count["boilerplate_hits"] += 1
        return None
    # 3) 独立页码
    if _is_standalone_page_number(line):
        count["boilerplate_hits"] += 1  # 页码归为 boilerplate 类删除
        return None
    # 4) 纯分隔符（无语义）→ 低信息块
    if _is_pure_separator(line):
        count["low_info_blocks_removed"] += 1
        return None
    # 5) 占位符标准化（可选，合同类启用）
    if normalize_placeholder:
        new_line, h = _normalize_placeholder_in_line(line)
        if h:
            count["placeholder_normalized"] += h
            return new_line
    return line


def _collapse_whitespace(text: str) -> tuple[str, int]:
    """压空白：CRLF→LF，连续空格/Tab→单空格，>=3 换行→2 换行，strip。返回(文本, 移除空行数估)。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除行尾空白
    text = "\n".join(part.rstrip() for part in text.split("\n"))
    text = re.sub(r"[ \t]+", " ", text)
    before = text.count("\n\n\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text, before


# =============================================================================
# PDF 跨页页眉/页脚检测（仅多页文档启用）
# =============================================================================
def _detect_repeated_boundaries(pages: list[list[str]]) -> set[str]:
    """
    返回在所有页面中、前/后 N 行内重复出现的标准化短文本集合（判定为页眉/页脚）。
    保守：要求 >= PDF_REPEATED_BOUNDARY_RATIO 比例页面在同一边界位置出现相同文本。
    """
    if len(pages) < PDF_MIN_PAGES_FOR_BOUNDARY:
        return set()
    n = PDF_BOUNDARY_SCAN_LINES
    candidates: list[dict[str, str]] = []
    for page in pages:
        head = [ln.strip() for ln in page[:n]]
        tail = [ln.strip() for ln in page[-n:]]
        candidates.append({"head": head, "tail": tail})

    repeated: set[str] = set()

    def _majority(seqs: list[list[str]]) -> set[str]:
        # 对每个位置 i，统计文本频率，超过比例阈值则标记
        out: set[str] = set()
        if not seqs:
            return out
        width = max((len(s) for s in seqs), default=0)
        for i in range(width):
            freq: dict[str, int] = {}
            for s in seqs:
                if i < len(s) and s[i]:
                    freq[s[i]] = freq.get(s[i], 0) + 1
            for tok, c in freq.items():
                if c / len(seqs) >= PDF_REPEATED_BOUNDARY_RATIO:
                    out.add(tok)
        return out

    head_texts = [c["head"] for c in candidates]
    tail_texts = [c["tail"] for c in candidates]
    repeated |= _majority(head_texts)
    repeated |= _majority(tail_texts)
    # 不把页面中间可能重复的正文标题误判：仅在边界集合中
    return repeated


# =============================================================================
# 公开入口
# =============================================================================
def _select_profile(parsed: ParsedDocument, profile: str | None) -> str:
    if profile and profile != "auto":
        return profile
    ext = (parsed.metadata.get("extension") or "").lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in (".md", ".txt"):
        # 派生自官方 HTML → 更严格网页清洗
        if parsed.metadata.get("derived") or parsed.metadata.get("derived_from") == "official_html":
            return "web_derived"
        return "generic"
    return "generic"


def clean_parsed_document(
    parsed: ParsedDocument,
    *,
    profile: str = "auto",
    enable_placeholder: bool | None = None,
    enable_pdf_boundary: bool = True,
) -> ParsedDocument:
    """
    确定性清洗 ParsedDocument，返回新的 ParsedDocument（不修改入参）。

    入参:
        parsed: parse_document 的输出。
        profile: "auto" | "pdf" | "docx" | "web_derived" | "generic"。
        enable_placeholder: 是否做合同占位符标准化；None 时由 profile 决定
                          （docx/pdf 开启，web_derived/generic 关闭以免误改正文）。
        enable_pdf_boundary: 是否启用 PDF 跨页页眉/页脚检测。
    返回:
        清洗后的 ParsedDocument（text/segments 更新，metadata 追加 cleaning 统计）。
        保留原始 metadata 全部字段（doc_id/title/source_file 等 identity 不变）。
    """
    chosen = _select_profile(parsed, profile)
    if enable_placeholder is None:
        enable_placeholder = chosen in ("docx", "pdf")

    raw_text = parsed.text or ""
    raw_chars = len(raw_text)

    count: dict[str, int] = {
        "boilerplate_hits": 0,
        "blank_lines_removed": 0,
        "low_info_blocks_removed": 0,
        "placeholder_normalized": 0,
    }

    # 按 segment（PDF=页；其他=整块）处理
    cleaned_segments: list[str] = []
    page_lines: list[list[str]] = []  # 供 PDF 边界检测

    for seg in parsed.segments or (parsed.text,):
        lines = seg.split("\n")
        kept: list[str] = []
        removed_junk_table_row = False
        for line in lines:
            if removed_junk_table_row and _is_markdown_separator_row(line):
                count["low_info_blocks_removed"] += 1
                removed_junk_table_row = False
                continue
            is_junk_table_row = _is_markdown_junk_row(line)
            res = _clean_line(line, count=count, normalize_placeholder=enable_placeholder)
            if res is None:
                removed_junk_table_row = is_junk_table_row
                if not line.strip():
                    count["blank_lines_removed"] += 1
                continue
            removed_junk_table_row = False
            kept.append(res)
        page_lines.append(kept)
        cleaned_segments.append("\n".join(kept))

    # PDF 跨页页眉/页脚检测（在 segment 层面移除边界重复行）
    if enable_pdf_boundary and chosen == "pdf":
        repeated = _detect_repeated_boundaries(page_lines)
        if repeated:
            n = PDF_BOUNDARY_SCAN_LINES
            new_segments: list[str] = []
            for idx, kept in enumerate(page_lines):
                head = kept[:n]
                tail = kept[-n:] if len(kept) >= n else []
                new_head = [ln for ln in head if ln.strip() not in repeated]
                new_tail = [ln for ln in tail if ln.strip() not in repeated]
                body = kept[n : len(kept) - n] if len(kept) >= n else kept[n:]
                merged = new_head + body + new_tail
                count["boilerplate_hits"] += (len(head) - len(new_head)) + (
                    len(tail) - len(new_tail)
                )
                new_segments.append("\n".join(merged))
            cleaned_segments = new_segments

    # 段落级空白压缩 + 重新 join
    joined = "\n\n".join(s for s in cleaned_segments if s.strip())
    clean_text, _ = _collapse_whitespace(joined)
    clean_chars = len(clean_text)
    removed_chars = max(raw_chars - clean_chars, 0)
    removed_ratio = round(removed_chars / raw_chars, 4) if raw_chars else 0.0
    normalized_sha = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    residual_hits, residual_patterns = _find_residual_boilerplate(clean_text)

    # 质量门
    reasons: list[str] = []
    status = "PASS"
    if removed_ratio >= REMOVED_RATIO_REVIEW_THRESHOLD:
        status = "NEEDS_REVIEW"
        reasons.append(f"removed_ratio={removed_ratio} >= {REMOVED_RATIO_REVIEW_THRESHOLD}")
    if raw_chars > 0 and clean_chars < MIN_CLEAN_CHARS_REVIEW:
        status = "NEEDS_REVIEW"
        reasons.append(f"clean_chars={clean_chars} < {MIN_CLEAN_CHARS_REVIEW}（正文可能为空）")
    if residual_hits > 0:
        status = "NEEDS_REVIEW"
        reasons.append(f"residual_boilerplate_hits={residual_hits}: {', '.join(residual_patterns)}")

    stats = CleaningStats(
        raw_chars=raw_chars,
        clean_chars=clean_chars,
        removed_chars=removed_chars,
        removed_ratio=removed_ratio,
        boilerplate_hits=count["boilerplate_hits"],
        placeholder_normalized=count["placeholder_normalized"],
        blank_lines_removed=count["blank_lines_removed"],
        low_info_blocks_removed=count["low_info_blocks_removed"],
        normalized_sha256=normalized_sha,
        quality_status=status,
        quality_reasons=reasons,
    )

    # 保留原始 identity，追加 cleaning 统计
    new_meta = dict(parsed.metadata)
    new_meta["cleaning"] = {
        "version": CLEANING_VERSION,
        "profile": chosen,
        "raw_chars": stats.raw_chars,
        "clean_chars": stats.clean_chars,
        "removed_chars": stats.removed_chars,
        "removed_ratio": stats.removed_ratio,
        "boilerplate_hits": stats.boilerplate_hits,
        "placeholder_normalized": stats.placeholder_normalized,
        "blank_lines_removed": stats.blank_lines_removed,
        "low_info_blocks_removed": stats.low_info_blocks_removed,
        "normalized_sha256": stats.normalized_sha256,
        "quality_status": stats.quality_status,
        "quality_reasons": stats.quality_reasons,
        "residual_boilerplate_hits": residual_hits,
        "residual_boilerplate_patterns": residual_patterns,
    }
    # normalize placeholder 标志，便于后续追溯
    new_meta["cleaning"]["placeholder_normalized_enabled"] = enable_placeholder

    return ParsedDocument(
        text=clean_text,
        segments=tuple(cleaned_segments),
        metadata=new_meta,
    )
