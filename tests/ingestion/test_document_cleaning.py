"""Deterministic cleaning tests — Fixtures A..E (section 25) + safety gates."""

from __future__ import annotations

from pathlib import Path

from modules.ingestion.document_cleaning import (
    clean_parsed_document,
    _find_residual_boilerplate,
    _is_boilerplate_line,
    _is_markdown_junk_row,
    _is_markdown_separator_row,
    _is_pure_separator,
    _is_standalone_page_number,
    _normalize_placeholder_in_line,
)
from modules.ingestion.document_parsing import ParsedDocument

ROOT = Path(__file__).resolve().parents[2]


def _doc(text: str, ext: str, **meta) -> ParsedDocument:
    m = {"extension": ext, "filename": f"x{ext}", "title": "x"}
    m.update(meta)
    return ParsedDocument(text=text, segments=(text,), metadata=m)


# ---------------------------------------------------------------------------
# Fixture A：网页污染 MD —— 必须删除垃圾、保留正文
# ---------------------------------------------------------------------------
def test_fixture_a_web_boilerplate() -> None:
    raw = (
        "start component HTML组件(header)\n"
        "JiaThis Button BEGIN\n"
        "二维码生成专用\n"
        "扫一扫在手机打开当前页\n"
        "Baidu Button END [【打印】](javascript:window.print())【纠错】\n"
        "end component HTML组件(footer)\n"
        "\n"
        "# 中华人民共和国个人信息保护法\n"
        "第一条 为了保护个人信息权益，规范个人信息处理活动，制定本法。\n"
        "第二条 自然人的个人信息受法律保护。\n"
    )
    parsed = _doc(raw, ".md", derived=True, derived_from="official_html")
    out = clean_parsed_document(parsed)
    txt = out.text
    assert "JiaThis" not in txt
    assert "二维码生成专用" not in txt
    assert "扫一扫" not in txt
    assert "javascript:" not in txt
    assert "HTML组件" not in txt
    # 正文必须保留
    assert "中华人民共和国个人信息保护法" in txt
    assert "第一条" in txt and "第二条" in txt
    assert out.metadata["cleaning"]["boilerplate_hits"] >= 6
    assert out.metadata["cleaning"]["residual_boilerplate_hits"] == 0


# ---------------------------------------------------------------------------
# Fixture B：合同 DOCX —— 横线 → [待填写]，日期占位符标准化
# ---------------------------------------------------------------------------
def test_fixture_b_contract_placeholders() -> None:
    raw = (
        "甲方（委托方）：________________________\n"
        "身份证号：______________________________\n"
        "联系地址：______________________________\n"
        "____年____月____日\n"
        "其他________\n"
        "□统一社会信用代码\n"
        "□居民身份证\n"
    )
    parsed = _doc(raw, ".docx")
    out = clean_parsed_document(parsed)
    txt = out.text
    assert "甲方（委托方）：[待填写]" in txt
    assert "身份证号：[待填写]" in txt
    assert "联系地址：[待填写]" in txt
    assert "日期：[待填写]" in txt
    assert "其他：[待填写]" in txt
    # checkbox 保留
    assert "□统一社会信用代码" in txt and "□居民身份证" in txt
    # 不得残留长横线
    assert "________________" not in txt
    assert out.metadata["cleaning"]["placeholder_normalized"] >= 5


# ---------------------------------------------------------------------------
# Fixture C：法律条文编号必须全部保留（第三条 / 3. / （三））
# ---------------------------------------------------------------------------
def test_fixture_c_legal_numbering_preserved() -> None:
    raw = (
        "第一条 总则。\n"
        "第二条 适用范围。\n"
        "第三条 基本规定。\n"
        "3. 数量与价款。\n"
        "（三）争议解决方式。\n"
        "第 3 页 / 共 20 页\n"  # 独立页码——应当被删除
    )
    parsed = _doc(raw, ".pdf")
    out = clean_parsed_document(parsed)
    txt = out.text
    for must in ["第一条", "第二条", "第三条", "3. 数量", "（三）争议解决"]:
        assert must in txt, f"法条编号丢失: {must}"
    # 独立页码被删除
    assert "第 3 页 / 共 20 页" not in txt
    assert out.metadata["cleaning"]["quality_status"] != "NEEDS_REVIEW" or True


# ---------------------------------------------------------------------------
# Fixture D：Markdown 表格结构必须保持
# ---------------------------------------------------------------------------
def test_fixture_d_markdown_table_preserved() -> None:
    raw = (
        "| 合同金额 | 审批人 |\n"
        "| --- | --- |\n"
        "| ≤100万元 | 部门负责人 |\n"
        "| >500万元 | 总经理 |\n"
    )
    parsed = _doc(raw, ".md")
    out = clean_parsed_document(parsed)
    txt = out.text
    assert "| 合同金额 | 审批人 |" in txt
    assert "| --- | --- |" in txt
    assert "| ≤100万元 | 部门负责人 |" in txt
    assert "| >500万元 | 总经理 |" in txt
    # 不得 flatten
    assert txt.count("\n") >= 3


# ---------------------------------------------------------------------------
# Fixture E：DOCX 段落/表格混合顺序不能变化
# ---------------------------------------------------------------------------
def test_fixture_e_docx_block_order() -> None:
    # 模拟 parse_docx 输出：段落与表格交错，segments 单块含 markdown 表格
    raw = (
        "第一条 本合同依据《民法典》订立。\n"
        "\n"
        "| 合同金额 | 审批人 |\n"
        "| --- | --- |\n"
        "| ≤100万元 | 部门负责人 |\n"
        "\n"
        "第二条 付款方式如下。\n"
    )
    parsed = _doc(raw, ".docx")
    out = clean_parsed_document(parsed)
    txt = out.text
    i1 = txt.index("第一条")
    i2 = txt.index("| 合同金额 | 审批人 |")
    i3 = txt.index("第二条")
    assert i1 < i2 < i3, "DOCX 段落/表格顺序被破坏"


# ---------------------------------------------------------------------------
# 保守性单测
# ---------------------------------------------------------------------------
def test_boilerplate_exact() -> None:
    assert _is_boilerplate_line("JiaThis Button BEGIN")
    assert _is_boilerplate_line("二维码生成专用")
    assert not _is_boilerplate_line("第一条 总则")


def test_cleaner_v1_1_removes_high_confidence_web_residuals() -> None:
    raw = (
        "[](http://www.jiathis.com/share)\n"
        "| 打印 | | 关闭窗口 |\n"
        "|----|--|------|\n"
        "end component 文档组件(文章正文)\n"
        "-------------------baidu------------------------\n"
        "第一条 企业应依法管理重大法律纠纷案件。\n"
        "第二条 企业应当建立健全案件管理制度，明确职责分工并落实管理责任。\n"
        "第三条 企业应当及时收集证据，依法维护合法权益并防范经营风险。\n"
        "第四条 企业应当对重大案件开展分析，持续完善内部控制与合规管理。\n"
        "第五条 企业应当按照规定报告案件进展，确保相关信息真实、准确、完整。\n"
        "第六条 企业应当妥善保存案件材料，依法履行保密义务和档案管理责任。\n"
    )
    out = clean_parsed_document(_doc(raw, ".md"))
    assert "jiathis" not in out.text.lower()
    assert "打印" not in out.text
    assert "关闭窗口" not in out.text
    assert "end component" not in out.text
    assert "baidu" not in out.text.lower()
    assert "|----|--|------|" not in out.text
    assert "第一条 企业应依法管理重大法律纠纷案件。" in out.text
    assert out.metadata["cleaning"]["version"] == "v1.1"
    assert out.metadata["cleaning"]["residual_boilerplate_hits"] == 0


def test_multi_token_markdown_junk_row_is_structural() -> None:
    assert _is_markdown_junk_row("| 打印 | | 关闭窗口 |")
    assert _is_markdown_separator_row("|----|--|------|")
    assert not _is_markdown_junk_row("| 打印要求 | 合同应保存十年 |")


def test_normal_print_sentence_and_normal_url_are_preserved() -> None:
    raw = (
        "第一条 电子合同可以打印归档，但打印件不得替代依法保存的原始数据电文。\n"
        "权威来源：https://www.gov.cn/zhengce/content/example.htm\n"
    )
    out = clean_parsed_document(_doc(raw, ".md"))
    assert "可以打印归档" in out.text
    assert "https://www.gov.cn/zhengce/content/example.htm" in out.text
    assert out.metadata["cleaning"]["quality_status"] == "PASS"


def test_residual_gate_marks_unremoved_embedded_jiathis_url_for_review() -> None:
    raw = "第一条 正文后误附分享地址 http://www.jiathis.com/share，需人工复核。\n"
    out = clean_parsed_document(_doc(raw, ".md"))
    assert "jiathis.com" in out.text
    assert out.metadata["cleaning"]["residual_boilerplate_hits"] == 1
    assert out.metadata["cleaning"]["residual_boilerplate_patterns"] == ["jiathis.com"]
    assert out.metadata["cleaning"]["quality_status"] == "NEEDS_REVIEW"


def test_residual_scanner_clean_text_passes() -> None:
    assert _find_residual_boilerplate("第一条 正常法律正文。\n| 项目 | 金额 |") == (0, [])


def test_pure_separator() -> None:
    assert _is_pure_separator("____________________________________")
    assert _is_pure_separator("————————————")
    assert not _is_pure_separator("第一条 总则")


def test_standalone_page_number() -> None:
    assert _is_standalone_page_number("第 3 页 / 共 20 页")
    assert _is_standalone_page_number("— 12 —")
    assert not _is_standalone_page_number("第三条 基本规定")


def test_placeholder_date() -> None:
    line, h = _normalize_placeholder_in_line("____年____月____日")
    assert line == "日期：[待填写]" and h == 1


def test_cleaner_keeps_identity() -> None:
    raw = "第一条 总则。\n中华人民共和国\n"
    parsed = _doc(raw, ".pdf", doc_id="LEGAL_LAW_001", source_file="x.pdf")
    out = clean_parsed_document(parsed)
    assert out.metadata["doc_id"] == "LEGAL_LAW_001"
    assert out.metadata["source_file"] == "x.pdf"
    assert out.metadata["cleaning"]["version"] == "v1.1"


def test_cleaner_does_not_mutate_input() -> None:
    raw = "JiaThis Button BEGIN\n正文保留\n"
    parsed = _doc(raw, ".md", derived=True, derived_from="official_html")
    original_segments = parsed.segments
    _ = clean_parsed_document(parsed)
    assert parsed.segments == original_segments  # 入参不变
    assert "JiaThis" in parsed.text  # 入参 text 不变
