"""引用核查：提取回答中的法条引用并逐条与本次检索证据比对（OpenSpec add-citation-check）。

定位：纵深防御的第二道防线——prompt 层证据规则（概率性防护）之外的机制性防护。
**观察模式（红线）**：只报告，绝不改写/拦截/重发答案；调用方必须 try/except fail-open。

三层结构（design D2/D3/D5/D6）：
1. 提取器（本文件上半部）：正则 + 中文数字解析，零 LLM 调用。
   覆盖形态：行内《法名》第X条 / 列表形态（"依据《法名》："后跟"第X条: 引文"）/
   第X条第Y款 / 第X条至第Y条（展开）/ 中文与阿拉伯数字互转 / 裸条号归属最近法名提及。
2. 验证器（本文件下半部）：四级状态判定 + 文本比对（规范化 → 包含 → difflib 相似度）。
   验证对象仅为本次证据（父文档 + 素材文本），绝不扩成全量语料（design D1）。
3. CitationReport：汇总计数 + 逐条明细，供 SSE citation_report 事件与审计日志消费。

已知限制（README 已声明）：v1 只覆盖《法名》+第X条族；"该法第五条"这类跨句指代不计（漏提
不产生错误状态——少报不少错）。
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =============================================================================
# 一、中文数字解析（任务 1.1）
# =============================================================================

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(s: str) -> int:
    """中文数字（一~九千九百九十九）或阿拉伯数字串 → int。

    覆盖实测形态：三十(30)、一百八十八(188)、一千二百六十(1260)、二百零五(205)、十(10)。
    """
    s = s.strip()
    if s.isdigit():
        return int(s)
    total, num = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (num or 1) * _CN_UNITS[ch]
            num = 0
        else:
            return -1  # 含非法字符：交由调用方按"解析失败"忽略
    return total + num


def int_to_cn(n: int) -> str:
    """int → 中文数字条号形态（30→"三十"，188→"一百八十八"，1260→"一千二百六十"）。

    用途：证据里的条号几乎都是中文写法（"第三十条"），把引用条号转成同形态再 find。
    覆盖 1~9999（法条条号足够；超出范围原样返回 str(n)）。
    """
    if not 1 <= n <= 9999:
        return str(n)
    units = [(1000, "千"), (100, "百"), (10, "十")]
    digits = "零一二三四五六七八九"
    out, rest = "", n
    started = False
    for value, unit in units:
        d, rest = divmod(rest, value)
        if d:
            out += ("" if d == 1 and not started and value == 10 else digits[d]) + unit
            started = True
        elif started and rest:
            out += "零"
    if rest or not out:
        out += digits[rest]
    return out


# 条号/款号/范围：第X条、第X条第Y款、第X条至第Y条（X/Y 为中文或阿拉伯数字）
# lookbehind 排除"本法/该法第X条"形态——那是引文内的**指代**（如"依照本法第三十八条"），
# 不是一条新引用；不排除的话引文里每提一次嵌套条号就会多报一条（漏提不产生错误状态，多提会污染计数）。
_ARTICLE_RE = re.compile(
    r"(?<!本法)(?<!该法)第(?P<art>[零〇一二三四五六七八九十百千]+|\d+)条"
    r"(?:第(?P<clause>[零〇一二三四五六七八九十]+|\d+)款)?"
)
_RANGE_RE = re.compile(
    r"第(?P<a>[零〇一二三四五六七八九十百千]+|\d+)条\s*[-—至到]\s*第?(?P<b>[零〇一二三四五六七八九十百千]+|\d+)条"
)
# 法名提及：《法名》
_LAW_RE = re.compile(r"《(?P<law>[^《》\n]{2,40})》")

# 简称归一用的国名前缀（design D5：全称 ↔ 简称双向匹配的中间桥）
_CN_PREFIX = "中华人民共和国"


@dataclass
class Citation:
    """一条结构化引用（提取器输出）。"""

    law: str              # 法名（书名号内原文）；空串 = 无名引用（素材条款场景，design D1）
    article: int          # 条号
    clause: int | None = None  # 款号（未提及为 None）
    quote: str = ""       # 该条引用的文本（条号之后到下一引用/段尾），供文本比对


def extract_citations(answer: str) -> list[Citation]:
    """从完整回答文本提取全部结构化引用（任务 1.2/1.3 的入口）。

    归属规则：按《法名》提及把回答切段，段内条号归属段头法名——
    等价于"裸条号归属最近一次法名提及"（向前最近）。首个法名提及之前的条号 law=""
    （若存在素材上下文，验证时按素材条款处理，见 spec 引用提取第 6 条）。
    """
    if not answer:
        return []

    # 先展开范围引用（"第三条至第五条"→逐条），避免与单条 pattern 抢匹配
    def _expand_range(m: re.Match) -> str:
        a, b = cn_to_int(m.group("a")), cn_to_int(m.group("b"))
        if a <= 0 or b <= 0 or b - a > 200:  # 防御：非法或超大范围不展开
            return m.group(0)
        return "、".join(f"第{int_to_cn(i)}条" for i in range(a, b + 1))

    text = _RANGE_RE.sub(_expand_range, answer)

    # 按《法名》提及切段：段 = (法名, 该提及到下一提及之间的文本)
    marks = list(_LAW_RE.finditer(text))
    segments: list[tuple[str, int, int]] = []  # (law, start, end)
    if marks:
        if marks[0].start() > 0:
            segments.append(("", 0, marks[0].start()))  # 首个法名之前的文本：无名段
        for i, m in enumerate(marks):
            seg_end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            segments.append((m.group("law"), m.end(), seg_end))

    citations: list[Citation] = []
    for law, seg_start, seg_end in segments:
        seg_text = text[seg_start:seg_end]
        matches = list(_ARTICLE_RE.finditer(seg_text))
        for j, m in enumerate(matches):
            article = cn_to_int(m.group("art"))
            if article <= 0:
                continue
            clause_raw = m.group("clause")
            clause = cn_to_int(clause_raw) if clause_raw else None
            if clause is not None and clause < 0:
                clause = None
            # 引文 = 条号之后到下一个条号（或段尾）的文本；
            # 截到第一个换行——列表形态（"- 第X条: 引文"逐行排列）的引文天然以行结束，
            # 不截会把列表尾部的总结句吞进引文（2.3 首跑实测：85 条吞了"建议保留证据…"句）
            q_end = matches[j + 1].start() if j + 1 < len(matches) else len(seg_text)
            raw_q = seg_text[m.end():q_end]
            nl = raw_q.find("\n")
            if nl >= 0:
                raw_q = raw_q[:nl]
            quote = raw_q.strip().strip("：:，,。；; \t")
            # 列举形态：条号后紧跟分隔符（、，和）= 枚举引用（"第三十条、第七十七条、第八十五条"），
            # 中间条号后的"引文"其实是分隔残留 → 置空（条号级验证）；探针②第六轮实测误报源之二
            if raw_q[:1] in ("、", "，", "和"):
                quote = ""
            # 行内指代形态（"根据第X条的规定"）：紧跟"的"说明后面是作者的话不是引文
            # → 不作文本比对（条号级验证即可）；真引文不会以"的"开头。
            # 同款："等"开头（顿号列举尾条号吞下的"等…"也是指代不是引文）
            if quote.startswith("的") or quote.startswith("等"):
                quote = ""
            citations.append(Citation(law=law, article=article, clause=clause, quote=quote))

    # 无任何法名提及：全篇为无名段（素材条款场景）
    if not marks:
        for m in _ARTICLE_RE.finditer(text):
            article = cn_to_int(m.group("art"))
            if article <= 0:
                continue
            citations.append(Citation(law="", article=article, quote=m.group(0)))
    return citations


# =============================================================================
# 二、验证器（任务 2.1/2.2）
# =============================================================================

STATUS_GROUNDED = "grounded"
STATUS_TEXT_MISMATCH = "text_mismatch"
STATUS_ARTICLE_MISSING = "article_missing"
STATUS_LAW_NOT_IN_EVIDENCE = "law_not_in_evidence"

# 页码残留（实测证据里有「—9—」这类 PDF 页码痕迹）与一切非文字数字符号
_PAGE_NOISE_RE = re.compile(r"—\s*\d+\s*—")
_NON_WORD_RE = re.compile(r"[^\w]")


def _normalize(s: str) -> str:
    """规范化：去页码痕迹 → 去全部标点空白（保留文字/数字/下划线）。"""
    return _NON_WORD_RE.sub("", _PAGE_NOISE_RE.sub("", s or ""))


def _law_keys(law: str) -> list[str]:
    """法名匹配键序列（design D5）：全称 → 去国名前缀。

    用于把回答法名与证据法名做双向兼容匹配（简称包含判断见 _laws_match）。
    """
    keys = [law]
    if law.startswith(_CN_PREFIX) and len(law) > len(_CN_PREFIX) + 1:
        keys.append(law[len(_CN_PREFIX):])
    return keys


def _laws_match(cited: str, evidence_law: str) -> bool:
    """两条法名是否指同一部法律。

    规则（design D5 + 探针①实测修正）：
    - 精确相等；或去「中华人民共和国」前缀后相等。
    - 正向包含：引用简称是证据全称的子串（「劳动合同法」⊂「…劳动合同法」）——安全。
    - 反向只允许**去前缀后精确相等**，不做子串包含：探针①实测「仲裁法」（证据）通过子串
      误认领了「劳动争议调解仲裁法」（引用）——后者是另一部更具体的法，子串共享不是同一部法。
      「全称引简称证据」的合法反向用例由"去前缀相等"覆盖（劳动合同法 ↔ …劳动合同法）。
    短名（≤2 字）不做包含匹配，防误伤。
    """
    if cited == evidence_law:
        return True
    c = cited[len(_CN_PREFIX):] if cited.startswith(_CN_PREFIX) else cited
    e = evidence_law[len(_CN_PREFIX):] if evidence_law.startswith(_CN_PREFIX) else evidence_law
    if not c or not e:
        return False
    if c == e:  # 去前缀后相等：全称↔简称的合法双向（含反向）
        return True
    if len(c) > 2 and c in e:  # 正向：引用简称 ⊂ 证据全称
        return True
    return False


def _evidence_law_names(evidence_text: str) -> list[str]:
    """从单篇证据提取法名候选：[Source Metadata] 头的 title/法名 + 正文书名号内名。"""
    names: list[str] = []
    meta = re.search(r"\[Source Metadata\](.*?)(?:\n\n|\n(?=第)|$)", evidence_text, re.S)
    if meta:
        for kv in re.finditer(r"(?:title|法名|来源)[^\n=]*?[=：:]\s*([^\n]+)", meta.group(1)):
            names.append(kv.group(1).strip())
    names.extend(m.group(1).strip() for m in _LAW_RE.finditer(evidence_text[:2000]))
    names.extend(m.group(1).strip() for m in _LAW_RE.finditer(evidence_text))
    return names


def _find_article_segment(evidence_text: str, article: int) -> tuple[int, int] | None:
    """在证据文本中定位"第{article}条"的正文段（条号 → 下一条号/文末）。

    兼容中文（第三十条）与阿拉伯（第30条）两种写法。返回 (段起, 段止) 偏移。
    """
    cn_form = f"第{int_to_cn(article)}条"
    for form in (cn_form, f"第{article}条"):
        i = evidence_text.find(form)
        if i >= 0:
            nxt = list(_ARTICLE_RE.finditer(evidence_text, i + len(form)))
            end = nxt[0].start() if nxt else len(evidence_text)
            return i, end
    return None


@dataclass
class CitationVerdict:
    """单条引用的核查结论（四级状态之一 + 命中证据片段）。"""

    law: str
    article: int
    clause: int | None
    status: str
    evidence_snippet: str = ""
    similarity: float = 1.0


@dataclass
class CitationReport:
    """整次核查的报告：逐条明细 + 汇总计数（D7 事件格式由此生成）。"""

    citations: list[CitationVerdict] = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return {
            "total": len(self.citations),
            "grounded": sum(v.status == STATUS_GROUNDED for v in self.citations),
            "text_mismatch": sum(v.status == STATUS_TEXT_MISMATCH for v in self.citations),
            "article_missing": sum(v.status == STATUS_ARTICLE_MISSING for v in self.citations),
            "law_not_in_evidence": sum(v.status == STATUS_LAW_NOT_IN_EVIDENCE for v in self.citations),
        }

    def to_event(self) -> dict:
        """SSE citation_report 事件体（design D7）。"""
        return {
            "type": "citation_report",
            "summary": self.summary,
            "citations": [
                {
                    "law": v.law,
                    "article": v.article,
                    "clause": v.clause,
                    "status": v.status,
                    "evidence_snippet": v.evidence_snippet,
                }
                for v in self.citations
            ],
        }


def verify_citations(
    answer: str,
    evidence: list[str] | str | None,
    threshold: float = 0.8,
) -> CitationReport:
    """对回答做引用核查（任务 2.1/2.2 的入口）。

    入参:
        answer: 完整回答文本
        evidence: 本次证据（父文档列表；素材路径下调用方已把素材文本 append 进来）。
                  None/空 → 提取仍进行，但所有带法名引用按 law_not_in_evidence 处理
                  （无名引用不判——没有素材文本就没有"素材条款"的验证对象）。
        threshold: 文本相似度阈值（CITATION_CHECK_TEXT_THRESHOLD，默认 0.8）
    返回:
        CitationReport（纯函数；绝不抛给调用方业务异常之外的信号，调用方仍需 try/except fail-open）
    """
    citations = extract_citations(answer)
    report = CitationReport()
    if not citations:
        return report

    ev_list: list[str] = []
    if isinstance(evidence, str):
        ev_list = [evidence]
    elif evidence:
        ev_list = [e for e in evidence if e]

    for c in citations:
        # ---- 无名引用（素材条款场景）：law 为空时直接对全部证据（素材文本在列表内）做条号级验证
        if not c.law:
            verdict = _verify_against_texts(c, ev_list, threshold)
            report.citations.append(verdict)
            continue

        # ---- 法名在证据中的存在性（简称归一双向，任务 1.3）----
        matched_texts = [
            t for t in ev_list
            if any(_laws_match(c.law, name) for name in _evidence_law_names(t))
        ]
        # 证据正文中未通过法名键匹配时，再按"回答法名（去前缀）直接出现于证据文本"兜底一次
        if not matched_texts:
            matched_texts = [t for t in ev_list if any(k in t for k in _law_keys(c.law) if k)]
        if not matched_texts:
            report.citations.append(CitationVerdict(
                law=c.law, article=c.article, clause=c.clause,
                status=STATUS_LAW_NOT_IN_EVIDENCE,
            ))
            continue

        verdict = _verify_against_texts(c, matched_texts, threshold)
        report.citations.append(verdict)
    return report


def _norm_sentences(s: str) -> list[str]:
    """按真实句读切分并逐句规范化（去页码/标点/空白）。

    刻意不按 \\n 切：PDF 证据的换行是**排版折行**不是句读（实测"劳动者可以解除\\n劳动合同"
    被从句中切开）——规范化去空白后折行自然接回，句读（。；;：:!?）才是可靠边界。

    用途：句级包含判断——回答常**跳款/跳句**引用（如只引第38条的（二）款，
    跳过（一）），整段包含必然失败，但逐句看每句都真实存在于段中 → 应判 grounded。
    """
    return [_normalize(part) for part in re.split(r"[。；;：:!！？?]", s or "") if _normalize(part)]


def _sentence_covered(norm_quote_sentences: list[str], norm_seg_sentences: list[str]) -> bool:
    """引文的每个规范化句都是证据段某句的子串（可跨句块命中，容忍跳款/跳句引用）。"""
    return all(
        any(qs in ss for ss in norm_seg_sentences)
        for qs in norm_quote_sentences
        if qs
    )


_CN_NUM_RUN_RE = re.compile(r"[零〇一二三四五六七八九十百千]+")


def _digits_covered(norm_quote: str, norm_seg: str) -> bool:
    """数字守卫：引文里的每个数字（含中文数字形态）都必须出现在证据段中。

    法律文本的命门是数字（金额/比例/期限）——「百分之五十」漂移成「百分之五」时
    相似度仍高达 0.95，必须靠数字守卫拦下（tasks 2.2 验收用例）。
    中文数字先统一转阿拉伯再比对（否则"五"是汉字，\\d+ 抓不到 → 守卫失灵，
    首跑实测教训）；两侧转换规则一致，普通词里的"一/十"等字两侧平衡不产生系统偏差。
    引文无数字时守卫自然放行（轻微改写交给滑窗相似度）。
    """
    def _to_arabic(m: re.Match) -> str:
        v = cn_to_int(m.group())
        return str(v) if v >= 0 else m.group()

    q_digits = re.findall(r"\d+", _CN_NUM_RUN_RE.sub(_to_arabic, norm_quote))
    if not q_digits:
        return True
    seg_digits = re.findall(r"\d+", _CN_NUM_RUN_RE.sub(_to_arabic, norm_seg))
    pool = list(seg_digits)
    for d in q_digits:
        if d in pool:
            pool.remove(d)  # 多重集合：段里两个"三十"只够覆盖引文里的两个
        else:
            return False
    return True


def _window_ratio(norm_quote: str, norm_seg: str) -> float:
    """滑窗最佳相似度：在证据段上滑动与引文等长的窗口取 max ratio。

    为什么不用整段比：回答常只引用条文中的一句（部分引用），短引文对长段的
    SequenceMatcher ratio 会被长度稀释（40 字引文 vs 150 字整段 ≈ 0.4 → 好引用被误杀）。
    窗口等长对等比 → 轻微改写（丢字/近义替换）可到 0.9+。
    防病态输入：段超 2000 字只取条号后前 2000（法条不可能更长）。
    """
    seg = norm_seg[:2000]
    q = norm_quote
    if not q or not seg:
        return 0.0
    if len(seg) <= len(q):
        return difflib.SequenceMatcher(None, q, seg).ratio()
    best = 0.0
    step = max(1, len(q) // 8)  # 窗口步长：长段时抽稀采样控成本（比例比对步长不敏感）
    for i in range(0, len(seg) - len(q) + 1, step):
        r = difflib.SequenceMatcher(None, q, seg[i : i + len(q)]).ratio()
        if r > best:
            best = r
            if best >= 0.999:
                break
    return best


def _verify_against_texts(c: Citation, texts: list[str], threshold: float) -> CitationVerdict:
    """条号级 + 文本级验证（texts = 已按法名筛过的证据，或无名引用的全部证据）。"""
    for t in texts:
        seg = _find_article_segment(t, c.article)
        if seg is None:
            continue
        raw_segment = t[seg[0]:seg[1]]
        snippet = raw_segment[:120].replace("\n", " ")
        norm_seg = _normalize(raw_segment)
        if not c.quote:
            # 无引文可比对（如行内"根据第三十条的规定"）：条号在证据中即判 grounded
            return CitationVerdict(c.law, c.article, c.clause, STATUS_GROUNDED, snippet)
        norm_quote = _normalize(c.quote)
        if norm_quote and norm_quote in norm_seg:
            return CitationVerdict(c.law, c.article, c.clause, STATUS_GROUNDED, snippet)
        # 句级包含：容忍跳款/跳句的部分引用（如只引第38条的（二）款，跳过（一））——
        # 整段包含会失败，但逐句看每句都真实存在于段中 → 不是改写，是合法摘引
        if norm_quote and _sentence_covered(_norm_sentences(c.quote), _norm_sentences(raw_segment)):
            return CitationVerdict(c.law, c.article, c.clause, STATUS_GROUNDED, snippet)
        # 数字守卫先于相似度（设计依据见 _digits_covered docstring）
        if norm_quote and not _digits_covered(norm_quote, norm_seg):
            return CitationVerdict(c.law, c.article, c.clause, STATUS_TEXT_MISMATCH, snippet, 0.0)
        ratio = _window_ratio(norm_quote, norm_seg)
        if ratio >= threshold:
            return CitationVerdict(c.law, c.article, c.clause, STATUS_GROUNDED, snippet, ratio)
        return CitationVerdict(c.law, c.article, c.clause, STATUS_TEXT_MISMATCH, snippet, ratio)
    # 所有候选证据中都没有该条号：法名在、条号不在
    return CitationVerdict(c.law, c.article, c.clause, STATUS_ARTICLE_MISSING)


def run_citation_check(
    answer: str,
    evidence: list[str] | str | None,
    threshold: float = 0.8,
) -> tuple[CitationReport | None, int]:
    """API 层入口（观察模式 + 审计日志，任务 7.1）。

    返回 (report, elapsed_ms)；report 为 None 表示无证据可核查（跳过事件——
    缓存命中等路径没有本次检索证据，发报告只会产生系统性误导）。
    任何内部异常向上抛出，由 API 层 try/except fail-open（红线 2）。
    """
    if not evidence:
        return None, 0
    t0 = time.perf_counter()
    report = verify_citations(answer, evidence, threshold)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    s = report.summary
    logger.info(
        "citation_report total=%d grounded=%d mismatch=%d missing=%d no_evidence=%d ms=%d",
        s["total"], s["grounded"], s["text_mismatch"],
        s["article_missing"], s["law_not_in_evidence"], elapsed_ms,
    )
    return report, elapsed_ms
