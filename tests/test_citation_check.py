"""引用核查（add-citation-check）的单元测试。

运行：pytest tests/test_citation_check.py -q

覆盖（对应 tasks 7 组中的组 1/组 2）：
1.1 中文数字解析（三十/30/一百八十八/一千二百六十/零〇/十/二百零五）与条/款/范围解析
1.2 引用提取（拖欠工资真实回答形态断言 5 条；行内+款号；阿拉伯数字；范围展开；裸条号归属）
1.3 简称归一（双向）
2.1 四级状态各一用例（grounded / text_mismatch / article_missing / law_not_in_evidence）
2.2 文本比对（轻微改写 grounded；数字漂移 mismatch）
2.3 ★真实样本回归：劳动合同法 PDF 解析文本作证据 → 5/5 grounded（核心验收锚点）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.rag.citation_check import (  # noqa: E402
    STATUS_ARTICLE_MISSING,
    STATUS_GROUNDED,
    STATUS_LAW_NOT_IN_EVIDENCE,
    STATUS_TEXT_MISMATCH,
    CitationReport,
    cn_to_int,
    extract_citations,
    int_to_cn,
    run_citation_check,
    verify_citations,
    _laws_match,
)

# ---------------------------------------------------------------------------
# 1.1 中文数字解析
# ---------------------------------------------------------------------------

def test_cn_to_int_variants():
    assert cn_to_int("三十") == 30
    assert cn_to_int("30") == 30
    assert cn_to_int("一百八十八") == 188
    assert cn_to_int("一千二百六十") == 1260
    assert cn_to_int("零") == 0
    assert cn_to_int("〇") == 0
    assert cn_to_int("十") == 10
    assert cn_to_int("二百零五") == 205
    assert cn_to_int("五") == 5


def test_int_to_cn_roundtrip():
    assert int_to_cn(30) == "三十"
    assert int_to_cn(188) == "一百八十八"
    assert int_to_cn(1260) == "一千二百六十"
    assert int_to_cn(5) == "五"
    assert int_to_cn(12) == "十二"
    assert int_to_cn(101) == "一百零一"
    for n in [3, 9, 10, 27, 30, 85, 101, 188, 205, 999, 1260]:
        assert cn_to_int(int_to_cn(n)) == n  # 往返自洽


# ---------------------------------------------------------------------------
# 1.2 引用提取（拖欠工资真实回答形态 —— 用户截图那轮的结构复刻）
# ---------------------------------------------------------------------------

# 模拟实测回答的真实结构：结构化依据段（"依据《法名》：" + 列表条目）。
# 引文取自 LEGAL_LAW_003 劳动合同法 PDF 的真实条文原文（保证 2.3 可断言 5/5 grounded）。
REAL_ARREARS_ANSWER = (
    "毕业生签订劳动合同后被拖欠工资，可以按以下步骤维权：\n"
    "1. 先与用人单位协商，要求说明拖欠原因并书面确认支付时间。\n"
    "2. 协商不成的，可以向劳动行政部门投诉，或依法解除劳动合同并主张经济补偿。\n\n"
    "依据《中华人民共和国劳动合同法》：\n"
    "- 第三十条: 用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。\n"
    "- 第三十八条: 用人单位有下列情形之一的，劳动者可以解除劳动合同:（二）未及时足额支付劳动报酬的；\n"
    "- 第四十六条: 有下列情形之一的，用人单位应当向劳动者支付经济补偿:（一）劳动者依照本法第三十八条规定解除劳动合同的；\n"
    "- 第四十七条: 经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算。\n"
    "- 第八十五条: 用人单位有下列情形之一的，由劳动行政部门责令限期支付劳动报酬、加班费或者经济补偿；逾期不支付的，责令用人单位按应付金额百分之五十以上百分之一百以下的标准向劳动者加付赔偿金:\n"
    "建议保留劳动合同、工资流水等证据，先协商后投诉，必要时申请劳动仲裁。\n"
)


def test_extract_list_form_real_answer_five_citations():
    """拖欠工资回答（实测形态）：提取出恰好 5 条，法名与条号全部正确。

    关键断言：第四十六条引文里嵌套的"依照本法第三十八条"**不得**产生第 6 条引用
    （"本法第X条"是指代不是新引用——lookbehind 排除）。
    """
    cites = extract_citations(REAL_ARREARS_ANSWER)
    assert len(cites) == 5, f"应提取 5 条，实际 {len(cites)}：{[(c.law, c.article) for c in cites]}"
    assert {c.article for c in cites} == {30, 38, 46, 47, 85}
    assert all(c.law == "中华人民共和国劳动合同法" for c in cites)
    # 引文非空且确实跟在条号后
    by_art = {c.article: c for c in cites}
    assert "及时足额支付劳动报酬" in by_art[30].quote
    assert "经济补偿按劳动者在本单位工作的年限" in by_art[47].quote


def test_extract_inline_with_clause():
    text = "根据《民法典》第一百八十八条第一款的规定，向人民法院请求保护民事权利的诉讼时效期间为三年。"
    cites = extract_citations(text)
    assert len(cites) == 1
    assert cites[0].law == "民法典"
    assert cites[0].article == 188
    assert cites[0].clause == 1


def test_extract_arabic_article_number():
    text = "《劳动合同法》第30条规定了劳动报酬的支付义务。"
    cites = extract_citations(text)
    assert len(cites) == 1
    assert cites[0].law == "劳动合同法"
    assert cites[0].article == 30  # 与"第三十条"等价


def test_extract_range_expansion():
    text = "《某法》第三条至第五条对此作了规定。"
    cites = extract_citations(text)
    assert [c.article for c in cites] == [3, 4, 5]
    assert all(c.law == "某法" for c in cites)


def test_extract_bare_article_belongs_to_nearest_law():
    """裸条号归属最近一次法名提及；首个法名之前的条号 law 为空（素材条款场景）。"""
    text = (
        "第八条 首先出现的裸条号没有法名可归属。\n"
        "依据《劳动法》：第八条 用人单位应当……\n"
        "另见《民法典》第一百八十八条的诉讼时效规定。"
    )
    cites = extract_citations(text)
    assert len(cites) == 3
    assert cites[0].law == "" and cites[0].article == 8      # 法名之前：无名
    assert cites[1].law == "劳动法" and cites[1].article == 8  # 归属最近提及
    assert cites[2].law == "民法典" and cites[2].article == 188


# ---------------------------------------------------------------------------
# 1.3 简称归一（双向）
# ---------------------------------------------------------------------------

def test_laws_match_both_directions():
    # 简称 → 全称
    assert _laws_match("劳动合同法", "中华人民共和国劳动合同法")
    # 全称 → 简称
    assert _laws_match("中华人民共和国劳动合同法", "劳动合同法")
    # 去前缀等价
    assert _laws_match("民法典", "中华人民共和国民法典")
    # 不同法律不误匹配
    assert not _laws_match("劳动合同法", "中华人民共和国民法典")
    assert not _laws_match("劳动合同法", "仲裁法")
    # 探针①实测回归：子串共享 ≠ 同一部法——《仲裁法》（证据）不得认领
    # 《劳动争议调解仲裁法》（引用）的条款（后者是另一部更具体的法）
    assert not _laws_match("劳动争议调解仲裁法", "中华人民共和国仲裁法")
    assert not _laws_match("中华人民共和国劳动争议调解仲裁法", "仲裁法")


def test_verify_short_name_hits_full_name_evidence():
    """回答用简称、证据用全称（含 Source Metadata 头的父文档形态）→ 进入条号级验证并 grounded。"""
    evidence = (
        "[Source Metadata]\ntitle=中华人民共和国劳动合同法\nsource_file=LEGAL_LAW_003_中华人民共和国劳动合同法.pdf\n\n"
        "第三十条 用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。"
    )
    report = verify_citations("《劳动合同法》第三十条：用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。", [evidence])
    assert report.summary["total"] == 1
    assert report.citations[0].status == STATUS_GROUNDED


# ---------------------------------------------------------------------------
# 2.1 四级状态（各一用例）
# ---------------------------------------------------------------------------

# 证据统一用生产形态：父文档带 [Source Metadata] 头（法名匹配依据，_with_source_metadata 注入）
_LABOR_LAW_SEG = (
    "[Source Metadata]\ntitle=中华人民共和国劳动合同法\nsource_file=LEGAL_LAW_003_中华人民共和国劳动合同法.pdf\n\n"
    "第三十条 用人单位应当按照劳动合同约定和国家规定，向\n\n—9—\n劳动者及时足额支付劳动报酬。\n"
    "用人单位拖欠或者未足额支付劳动报酬的，劳动者可以依法\n向当地人民法院申请支付令。\n"
    "第八十五条 用人单位有下列情形之一的，由劳动行政部门\n责令限期支付劳动报酬；逾期不支付的，责令\n"
    "用人单位按应付金额百分之五十以上百分之一百以下的标准向\n劳动者加付赔偿金:\n"
    "第九十条 劳动者违反本法规定解除劳动合同，给用人单位造成损失的，应当承担赔偿责任。"
)


def _with_header(body: str) -> str:
    """裸条文段 → 生产形态证据（带 Source Metadata 头）。"""
    return (
        "[Source Metadata]\ntitle=中华人民共和国劳动合同法\nsource_file=LEGAL_LAW_003_中华人民共和国劳动合同法.pdf\n\n"
        + body
    )


def test_status_grounded():
    text = "依据《劳动合同法》：第三十条: 用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。"
    r = verify_citations(text, [_LABOR_LAW_SEG])
    assert r.citations[0].status == STATUS_GROUNDED
    assert "第三十条" in r.citations[0].evidence_snippet


def test_status_article_missing():
    """证据只到第九十条，引用第一百零一条 → article_missing（最高风险状态）。"""
    text = "依据《劳动合同法》：第一百零一条: 某某 fictional 条文内容。"
    r = verify_citations(text, [_LABOR_LAW_SEG])
    assert r.citations[0].status == STATUS_ARTICLE_MISSING


def test_status_law_not_in_evidence():
    """《劳动争议调解仲裁法》不在证据中（语料外探针场景）→ law_not_in_evidence。"""
    text = "根据《劳动争议调解仲裁法》第二十七条，仲裁时效期间为一年。"
    r = verify_citations(text, [_LABOR_LAW_SEG])
    assert r.citations[0].status == STATUS_LAW_NOT_IN_EVIDENCE


def test_status_text_mismatch_number_drift():
    """数字漂移（百分之五十/一百 → 百分之五/十）：最高危改写，必须判 mismatch。"""
    text = "依据《劳动合同法》：第八十五条: 逾期不支付的，责令用人单位按应付金额百分之五以上百分之十以下的标准向劳动者加付赔偿金。"
    r = verify_citations(text, [_LABOR_LAW_SEG])
    assert r.citations[0].status == STATUS_TEXT_MISMATCH


def test_report_summary_and_event_shape():
    """汇总计数与 D7 事件结构。"""
    text = "《劳动合同法》第三十条: 用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。\n《劳动争议调解仲裁法》第二十七条: 时效一年。"
    r = verify_citations(text, [_LABOR_LAW_SEG])
    s = r.summary
    assert s["total"] == 2
    assert s["grounded"] == 1
    assert s["law_not_in_evidence"] == 1
    ev = r.to_event()
    assert ev["type"] == "citation_report"
    assert ev["summary"] == s
    assert {"law", "article", "clause", "status", "evidence_snippet"} <= set(ev["citations"][0])


# ---------------------------------------------------------------------------
# 2.2 文本比对（轻微改写 vs 数字漂移）
# ---------------------------------------------------------------------------

def test_text_compare_light_rewrite_grounded():
    """轻微改写（"都有权举报"→"有权举报"级别的丢字/替换）仍判有据。"""
    evidence = _with_header(
        "第三十二条 劳动者对危害生命安全和身体健康的劳动条件，有权对用人\n单位提出批评、检举和控告。\n"
        "第三十三条 用人单位与劳动者约定服务期的条款。"
    )
    answer = "《劳动合同法》第三十二条: 劳动者对危害生命安全和身体健康的劳动条件，都有权对用人单位提出批评、检举和控告。"
    r = verify_citations(answer, [evidence])
    assert r.citations[0].status == STATUS_GROUNDED, f"轻微改写应判 grounded，实际 {r.citations[0].status} sim={r.citations[0].similarity}"


def test_text_compare_page_noise_ignored():
    """证据中的 PDF 页码残留（—9—）与换行不得影响包含判断。"""
    evidence = _with_header(
        "第三十条 用人单位应当\n\n—9—\n按照劳动合同约定和国家规定，向\n劳动者及时足额支付劳动报酬。"
    )
    answer = "《劳动合同法》第三十条: 用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。"
    r = verify_citations(answer, [evidence])
    assert r.citations[0].status == STATUS_GROUNDED


def test_empty_quote_inline_citation_is_grounded():
    """行内引用没抓到引文（"根据第三十条的规定"）：条号在证据中即 grounded。"""
    evidence = _with_header("第三十条 用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。")
    answer = "根据《劳动合同法》第三十条的规定，单位必须按时足额发工资。"
    r = verify_citations(answer, [evidence])
    assert r.citations[0].status == STATUS_GROUNDED


def test_enumerated_citations_tail_not_swallowing_prose():
    """顿号列举（"第三十条、第七十七条、第八十五条等维权"）：尾条号吞的"等…"是指代不是引文。

    探针②第六轮实测：列举尾条号把"等维权；…"建议句吞成引文 → 误判 text_mismatch。
    修复后：以"等"开头的引文视为指代（quote 置空），条号级验证。
    """
    evidence = _with_header(
        "第三十条 用人单位应当按照劳动合同约定支付劳动报酬。\n"
        "第七十七条 劳动者合法权益受到侵害的，有权要求有关部门依法处理。\n"
        "第八十五条 逾期不支付劳动报酬的，责令加付赔偿金。"
    )
    answer = "可依据《劳动合同法》第三十条、第七十七条、第八十五条等维权；建议先书面催告。"
    r = verify_citations(answer, [evidence])
    assert r.summary["total"] == 3
    assert r.summary["grounded"] == 3, (
        f"列举引用应全部条号级 grounded，实际 {[ (v.article, v.status) for v in r.citations]}"
    )


def test_partial_quote_skipping_items_is_grounded():
    """跳款引用（只引第38条的（二）款，跳过（一））→ 句级包含 → grounded（2.3 首跑暴露的真实形态）。"""
    evidence = _with_header(
        "第三十八条 用人单位有下列情形之一的，劳动者可以解除\n劳动合同:\n"
        "（一）未按照劳动合同约定提供劳动保护或者劳动条件的；\n"
        "（二）未及时足额支付劳动报酬的；\n"
        "（三）未依法为劳动者缴纳社会保险费的；\n"
        "第三十九条 其他条款内容。"
    )
    answer = "《劳动合同法》第三十八条: 用人单位有下列情形之一的，劳动者可以解除劳动合同:（二）未及时足额支付劳动报酬的；"
    r = verify_citations(answer, [evidence])
    assert r.citations[0].status == STATUS_GROUNDED, f"跳款引用应判 grounded，实际 {r.citations[0].status}"


# ---------------------------------------------------------------------------
# 2.3 ★ 真实样本回归（核心验收锚点：把用户的人工核对固化成自动测试）
# ---------------------------------------------------------------------------

def test_real_sample_arrears_answer_five_of_five_grounded():
    """劳动合同法 PDF 解析文本作证据 + 拖欠工资回答（真实形态 5 条引用）→ 5/5 grounded。

    等价于把 2026-09-19 用户人工抽检（截图逐条对原文）固化为自动断言。
    注意 PDF 解析文本含换行与「—9—」「—14—」页码残留——全部由规范化吸收。
    """
    from modules.ingestion.document_parsing import parse_document

    pdf = Path(r"D:\xiaoyi\Legal_System\data_corpus\LEGAL_LAW_003_中华人民共和国劳动合同法.pdf")
    if not pdf.exists():
        import pytest
        pytest.skip("劳动合同法 PDF 不在本机（CI 环境跳过真实样本回归）")
    doc = parse_document(pdf)

    report = verify_citations(REAL_ARREARS_ANSWER, [doc.text])
    s = report.summary
    assert s["total"] == 5, f"应提取 5 条，实际 {s['total']}"
    assert s["grounded"] == 5, (
        f"5/5 grounded 验收失败：grounded={s['grounded']} "
        f"mismatch={s['text_mismatch']} missing={s['article_missing']} "
        f"no_evidence={s['law_not_in_evidence']}; "
        f"明细={[(v.law, v.article, v.status, round(v.similarity, 3)) for v in report.citations]}"
    )


# ---------------------------------------------------------------------------
# run_citation_check（API 层入口语义：无证据 → None；审计日志可打）
# ---------------------------------------------------------------------------

def test_run_citation_check_no_evidence_returns_none():
    report, ms = run_citation_check("任何回答", None)
    assert report is None
    report, ms = run_citation_check("任何回答", [])
    assert report is None


def test_run_citation_check_returns_report_and_ms():
    report, ms = run_citation_check(
        REAL_ARREARS_ANSWER[:80],
        [_with_header("第三十条 用人单位应当按照劳动合同约定支付劳动报酬。")],
    )
    assert isinstance(report, CitationReport)
    assert ms >= 0


# ---------------------------------------------------------------------------
# 3.1 配置字段与默认值
# ---------------------------------------------------------------------------

def test_citation_settings_defaults():
    from modules.core.config import get_settings

    s = get_settings()
    assert s.citation_check_enabled is False            # 默认关（红线 4：关闭=SSE 与现状逐字节一致）
    assert s.citation_check_text_threshold == 0.8


# ---------------------------------------------------------------------------
# 3.2 pipeline evidence_out（None=逐字节不变；成功收集/中断不写）
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from langchain_core.messages import AIMessageChunk  # noqa: E402

from modules.rag.pipeline import RagPipeline  # noqa: E402


def _fake_chunk(text: str) -> AIMessageChunk:
    return AIMessageChunk(content=text)


class _FakeLLM:
    """替身 LLM：astream 产出固定分片；可配置中途抛错。"""

    def __init__(self, pieces: list[str], fail_at: int | None = None) -> None:
        self._pieces = pieces
        self._fail_at = fail_at  # 第几个分片产出时抛错（None = 不失败）

    async def astream(self, messages):  # noqa: ANN001
        for i, p in enumerate(self._pieces):
            if self._fail_at == i:
                raise RuntimeError("boom during stream")
            yield _fake_chunk(p)


class _FakeCache:
    async def set_json(self, key, obj, ttl):  # noqa: ANN001
        self.stored = (key, obj)

    async def get_json(self, key):  # noqa: ANN001
        return None


class _FakeSem:
    async def store(self, *a, **k):  # noqa: ANN002, ANN003
        pass


def _bare_pipeline(llm) -> RagPipeline:
    """轻量构造：绕过 __init__（Milvus/模型等重依赖），只填 _rag_stream_llm 用到的属性。"""
    p = RagPipeline.__new__(RagPipeline)
    p.settings = get_settings_shallow()
    p._llm = lambda: llm
    p._cache = _FakeCache()
    p._sem = _FakeSem()

    async def _scope(q, uid):  # noqa: ANN001
        return ("scope", 0)

    p._cache_scope = _scope
    return p


def get_settings_shallow():  # noqa: ANN001
    from modules.core.config import get_settings
    return get_settings()


def test_rag_stream_llm_evidence_out_none_is_noop():
    """evidence_out=None（默认）：不收集，流式输出正常——红线 5（行为逐字节不变的参数面）。"""
    p = _bare_pipeline(_FakeLLM(["你好", "，世界"]))
    out: list[str] = []
    async def run():
        async for piece in p._rag_stream_llm("q", ["证据A"]):
            out.append(piece)
    asyncio.run(run())
    assert out == ["你好", "，世界"]  # 输出与不传参数完全一致


def test_rag_stream_llm_evidence_out_collects_on_success():
    """传入 list：流完整成功后 append 本次 contexts。"""
    p = _bare_pipeline(_FakeLLM(["答", "案"]))
    ev: list[str] = []
    out: list[str] = []
    async def run():
        async for piece in p._rag_stream_llm("q", ["证据A", "证据B"], evidence_out=ev):
            out.append(piece)
    asyncio.run(run())
    assert out == ["答", "案"]
    assert ev == ["证据A", "证据B"]  # 成功路径：拿到 contexts


def test_rag_stream_llm_evidence_out_not_written_on_abort():
    """中途异常（已产出分片后）：流正常收尾（不重试），evidence_out 不写入（无可靠证据）。"""
    p = _bare_pipeline(_FakeLLM(["部分答案", "永远到不了"], fail_at=1))  # 第 2 片产出前抛错
    ev: list[str] = []
    out: list[str] = []
    async def run():
        async for piece in p._rag_stream_llm("q", ["证据A"], evidence_out=ev):
            out.append(piece)
    asyncio.run(run())
    assert out == ["部分答案"]          # 已产出的分片照常送达
    assert ev == []                     # 中断路径：绝不写入


def test_rag_stream_llm_evidence_out_with_attachment_context():
    """素材模式（attachment_context 非空）：不写缓存但证据照常收集（素材路径核查需要）。"""
    p = _bare_pipeline(_FakeLLM(["答"]))
    ev: list[str] = []
    async def run():
        async for _ in p._rag_stream_llm("q", ["证据A"], attachment_context="素材文本", evidence_out=ev):
            pass
    asyncio.run(run())
    assert ev == ["证据A"]
    assert not hasattr(p._cache, "stored") or p._cache.stored is None  # 素材模式未写缓存


# ---------------------------------------------------------------------------
# 4.1/4.3 API 接线（事件时序 / 开关关闭零影响 / fail-open / 素材路径证据）
# ---------------------------------------------------------------------------

import json  # noqa: E402

import modules.core.config as _cfg  # noqa: E402
from backend.app.api import chat as chat_api  # noqa: E402
from backend.app.schemas import ChatRequest  # noqa: E402


class _FakeStreamPipeline:
    """替身 RagPipeline：stream_chat 产出固定分片并记录收到的 evidence_out 参数。"""

    def __init__(self, pieces: list[str], contexts: list[str] | None = None) -> None:
        self._pieces = pieces
        self._contexts = contexts or ["[Source Metadata]\ntitle=中华人民共和国劳动合同法\n\n第三十条 用人单位应当支付劳动报酬。"]
        self.seen_evidence_out = "UNSET"

    async def stream_chat(self, question, user_external_id=None, attachment_context="", evidence_out=None):  # noqa: ANN001
        self.seen_evidence_out = evidence_out
        if evidence_out is not None:
            evidence_out.extend(self._contexts)  # 模拟成功路径收集
        for p in self._pieces:
            yield p


def _collect_sse(resp) -> list[dict]:  # noqa: ANN001
    """把 StreamingResponse 的 SSE 文本行解成事件对象列表（保留 [DONE] 标记）。"""
    events: list[dict] = []
    async def run():
        async for chunk in resp.body_iterator:
            for line in chunk.splitlines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    events.append({"__done__": True})
                else:
                    events.append(json.loads(payload))
    asyncio.run(run())
    return events


def test_chat_stream_event_order_chunk_report_done():
    """4.1 时序：开关开 → …chunk… → citation_report → [DONE]。"""
    s = _cfg.get_settings()
    orig = (s.citation_check_enabled, s.citation_check_text_threshold)
    s.citation_check_enabled, s.citation_check_text_threshold = True, 0.8
    try:
        answer = "依据《中华人民共和国劳动合同法》：第三十条: 用人单位应当支付劳动报酬。"
        fake = _FakeStreamPipeline([answer[:10], answer[10:]])
        resp = asyncio.run(chat_api.chat_stream(ChatRequest(message="q"), pipeline=fake))
        events = _collect_sse(resp)
        types = [k for e in events for k in (("__done__",) if "__done__" in e else ("chunk",) if "chunk" in e else ("citation_report",) if e.get("type") == "citation_report" else ("other",))]
        assert "chunk" in types and "citation_report" in types and "__done__" in types
        assert types.index("citation_report") > types.index("chunk")          # 报告在首个 chunk 之后
        assert types.index("__done__") > types.index("citation_report")       # [DONE] 在报告之后
        report_ev = [e for e in events if e.get("type") == "citation_report"][0]
        assert report_ev["summary"]["total"] == 1
        assert report_ev["summary"]["grounded"] == 1
    finally:
        s.citation_check_enabled, s.citation_check_text_threshold = orig


def test_chat_stream_disabled_no_event_and_evidence_none():
    """4.3 开关关：pipeline 收到 evidence_out=None（不收集）、SSE 无 citation_report——与改造前一致。"""
    s = _cfg.get_settings()
    orig = s.citation_check_enabled
    s.citation_check_enabled = False
    try:
        fake = _FakeStreamPipeline(["普通回答，无引用。"])
        resp = asyncio.run(chat_api.chat_stream(ChatRequest(message="q"), pipeline=fake))
        events = _collect_sse(resp)
        assert fake.seen_evidence_out is None                          # 未向 pipeline 请求收集
        assert not any(e.get("type") == "citation_report" for e in events)  # 无核查事件
        assert events[-1] == {"__done__": True}                        # 流照常 [DONE] 收尾
    finally:
        s.citation_check_enabled = orig


def test_chat_stream_checker_crash_fail_open():
    """4.1 红线 2：核查器内部异常 → 无事件、流仍 [DONE] 正常收尾（fail-open）。"""
    s = _cfg.get_settings()
    orig = s.citation_check_enabled
    s.citation_check_enabled = True
    try:
        fake = _FakeStreamPipeline(["依据《中华人民共和国劳动合同法》：第三十条: 用人单位应当支付劳动报酬。"])

        async def _boom(*a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("checker internal crash")

        orig_checker = chat_api.run_citation_check
        chat_api.run_citation_check = _boom
        try:
            resp = asyncio.run(chat_api.chat_stream(ChatRequest(message="q"), pipeline=fake))
            events = _collect_sse(resp)
            assert not any(e.get("type") == "citation_report" for e in events)
            assert events[-1] == {"__done__": True}  # 答案流完整收尾，未受核查故障影响
        finally:
            chat_api.run_citation_check = orig_checker
    finally:
        s.citation_check_enabled = orig


def test_attachment_path_evidence_includes_material_text():
    """4.2 素材路径：证据 = 检索上下文 + 素材文本（合同条款对素材验证，spec「素材条款引用」）。"""
    import modules.rag.citation_check as cc
    from modules.rag.attachment import ExtractedAttachment

    # 素材文本：合同第十一条（无名引用场景——回答裸引"第十一条"归属素材条款）
    material = "第十一条 违约责任：乙方逾期交付的，每日按合同总价的千分之一支付违约金。\n第十二条 争议解决。"
    evidence_from_pipeline = ["[Source Metadata]\ntitle=中华人民共和国劳动合同法\n\n第三十条 用人单位应当支付劳动报酬。"]
    answer = "该合同的违约责任见第十一条：乙方逾期交付的，每日按合同总价的千分之一支付违约金。"

    # 组装 attachment 路径的证据列表（API 层语义：pipeline 证据 + append 素材文本）
    evidence = list(evidence_from_pipeline)
    evidence.append(material)
    report = cc.verify_citations(answer, evidence)
    s = report.summary
    assert s["total"] == 1
    assert s["grounded"] == 1, f"素材条款应 grounded，实际 {[(v.law, v.article, v.status) for v in report.citations]}"
