"""
HTML Parser V2 — ISOLATED PRODUCTION-QUALITY IMPLEMENTATION (NOT YET INTEGRATED)
=================================================================================
Scope: Formal Parser module, output = Frozen ParsedDocument V2 Contract.

FREEZE DECISIONS (migrated from A2 De-leak POC):
  * DOM API         = BeautifulSoup / bs4
  * Parser backend  = html.parser   (EXPLICIT single backend; NO lxml fallback)
  * GOLD DEPENDENCY = 0 (this module never sees gold/expected/fixture manifests)
  * NEW DEPENDENCY  = 0 (bs4 + stdlib html.parser only)
  * OUTPUT          = modules.ingestion.parsed_document_v2.ParsedDocument EXCLUSIVELY
                      (Contract frozen; this module never redefines Block/BlockType/TableData)

GENERIC CAPABILITIES MIGRATED FROM A2 POC:
  1. Malformed HTML handling   (html.parser lenient + soup-level decompose safety try/except)
  2. Main-content candidate discovery  (main/article/role=main/id/class patterns + fallback scan)
  3. Generic candidate scoring  (length/heading/para density - link density + legal patterns)
  4. Pagination / disjoint-container union  (generic start+end completeness)
  5. Flat-DOM fallback  (if named candidates fail → long-text container scan)
  6. DOM-based title extraction  (<title> suffix strip → <h1> → legal-title regex)
  7. HTML <meta>/<time> metadata extraction  (publish_date, source_org)
  8. Semantic block extraction  (HEADING/PARAGRAPH/LIST/TABLE/UNKNOWN preserving unknown)

PARSER vs CLEANER BOUNDARY (enforced strictly here):
  ✅ Parser DELETEs — script, style, nav/footer/header (with content-marker safety),
                      containers id/class matching nav keywords with <400 chars.
  ✅ Parser KEEPs  — uncertain boilerplate / share phrases mixed with body / dubious UI
                      leftovers. When in doubt, emit as UNKNOWN and add warning.
  ❌ Parser NEVER deletes body-proximal content just to force noise to zero.
  ❌ This module is NOT a Cleaner regex engine.

TODO: Production integration + __init__.py entry will be a follow-up task once
isolated review passes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# =============================================================================
# FROZEN CONTRACT — import ONLY from official module, do NOT copy types.
# =============================================================================
from modules.ingestion.parsed_document_v2 import (
    Block,
    BlockType,
    ContractViolation,
    LegacyAdapter,
    LegacyParsedDocument,
    ParsedDocument,
    Provenance,
    TableCell,
    TableData,
)


# =============================================================================
# Public API constants
# =============================================================================
PARSER_NAME: str = "html_parser_v2"
PARSER_VERSION: str = "2.0.0-isolated"
PARSER_DOM_API: str = "BeautifulSoup/bs4"
PARSER_BACKEND: str = "html.parser"
PARSER_NEW_DEPENDENCIES: int = 0  # bs4 (already existing) + stdlib html.parser only


# =============================================================================
# Generic heuristic pattern library — ALL domain conventions, ZERO fixture id.
# =============================================================================
_NAV_KEYWORDS_CLASS_ID = [
    "nav", "menu", "sidebar", "share", "breadcrumb", "crumbs",
    "footer", "header", "topbar", "toolbar", "widget", "weixin",
    "weibo", "social", "print", "comment", "login", "register",
    "search-box", "search", "pagination", "pager", "related",
    "hotnews", "notice", "banner", "popup", "float", "advert",
    "copyright", "icp", "beian", "sponsor", "operator",
]
_MAIN_CONTENT_SPEC = [
    {"tag": "main"},
    {"tag": "article"},
    {"role": "main"},
    {"id_re": r"(?i)^(UCAP-CONTENT|UCAP_CONTENT|pages_content)$"},
    {"class_re": r"(?i)(^|\s)(pages_content|UCAP-CONTENT|zhengwen|content_l|article-content|news-content)($|\s)"},
    {"class_re": r"(?i)(article[-_ ]?content|article[-_ ]?body|content[-_ ]?article|text[-_ ]?content|news[-_ ]?content|detail[-_ ]?content|view[-_ ]?content|show[-_ ]?content|post[-_ ]?content|page[-_ ]?content|content|article_body|articlebody|content_body|text_body|news_body|detail_body|view_body|show_body|post_body|TRS_Editor|zoom|contentbox|content_box|article_box|news_box|text_box)"},
    {"id_re":    r"(?i)(article[-_ ]?content|article[-_ ]?body|content[-_ ]?article|text[-_ ]?content|news[-_ ]?content|detail[-_ ]?content|view[-_ ]?content|show[-_ ]?content|post[-_ ]?content|page[-_ ]?content|content|article_body|articlebody|content_body|text_body|news_body|detail_body|view_body|show_body|post_body|TRS_Editor|zoom|contentbox|content_box|article_box|news_box|text_box)"},
]
_LIKELY_CONTENT_ID_RE = re.compile(
    r"(?i)(UCAP-CONTENT|TRS_Editor|pages_content|zhengwen|content_text|article[-_ ]?content|zoom)"
)
_LIKELY_CONTENT_CL_RE = re.compile(
    r"(?i)(pages_content|UCAP-CONTENT|TRS_Editor|zhengwen|zoom|article[-_ ]?content|content[-_ ]?article|text[-_ ]?content|news[-_ ]?content|detail[-_ ]?content|view[-_ ]?content|show[-_ ]?content|content_body|article_body|news_body|articlecontent|newscontent|contentbox)"
)
_PAGINATION_TAIL_RE = re.compile(r"(上一页|下一页|第\s*\d+\s*页|共\s*\d+\s*页|^\s*\d+\s+\d+\s+\d+\s*.*下一页)", re.M)
# Generic Chinese legal structure conventions (NOT fixture gold answers).
# These hold for all PRC laws/regulations/interpretations on gov.cn / cac / spcs courts.
_LAW_CHAPTER_RE = re.compile(r"第[一二三四五六七八九十百零〇\d]+章")
_LAW_ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇\d]+条")
_FIRST_CHAPTER_RE = re.compile(r"第一章|第1章|总则")
_FIRST_ARTICLE_RE = re.compile(r"第一条|第1条")
_LAW_ENFORCEMENT_END_RE = re.compile(
    r"(第[一二三四五六七八九十百零〇\d]+条.*?[本]?法自\d{4}年\d{1,2}月\d{1,2}日起施行"
    r"|本(办法|条例|规定|细则|解释).*?自\d{4}年\d{1,2}月\d{1,2}日起施行)"
)
_GENERIC_LEGAL_TITLE_RE = re.compile(
    r"(中华人民共和国[\u4e00-\u9fa5]{2,20}(法|条例|办法|规定|细则|解释|通则))"
    r"|(最高人民法院.*?(解释|规定|案例|答复))"
)
_SITE_SUFFIX_RE = re.compile(
    r"\s*[-_—–|／/]\s*("
    r"中国政府网|中央人民政府|中华人民共和国中央人民政府|国务院|国务院办公厅|"
    r"最高人民法院|中华人民共和国最高人民法院|最高人民检察院|"
    r"国家互联网信息办公室|中央网络安全和信息化委员会办公室|中央网信办|.*网信办|"
    r"全国人大|全国人大常委会|国资委|国务院国有资产监督管理委员会|CAC|"
    r"司法部|财政部|工业和信息化部|公安部|住房和城乡建设部|"
    r"中国人大网|中国法院网|人民法院新闻传媒总社|中国律师网|中国普法网|中国长安网|"
    r"人民网|新华网|中新网|发布系统|发布平台|官方网站|官网|发布$"
    r")\s*$"
)
_NAV_CLASS_ID_RE = re.compile(
    r"(?i)(^|[^a-z0-9])(" + "|".join(_NAV_KEYWORDS_CLASS_ID) + r")([^a-z0-9]|$)"
)
_META_TAG_CANDIDATES = {
    "publish_date": ["pubdate", "publishdate", "firstpublished", "publish-date",
                     "publish_date", "date", "created", "issued",
                     "article:published_time", "og:article:published_time"],
    "source_org":   ["source", "author", "copyright", "site-name", "publisher",
                     "application-name", "og:site_name", "article:author",
                     "og:author"],
}


# =============================================================================
# Internal helpers: candidate scoring (gold-free).
# =============================================================================
@dataclass(frozen=True)
class _CandidateFeatures:
    length: int
    n_headings: int
    n_paragraphs: int
    link_density: float
    short_links: int
    chapter_hits: int
    article_hits: int
    has_first_chapter_front: bool
    has_first_article_front: bool
    start_hit: bool
    end_hit: bool
    tail_penalty: bool


def _is_likely_content_marker(el: Any) -> bool:
    from bs4 import Tag
    if not isinstance(el, Tag):
        return False
    try:
        i = el.get("id") or ""
        c = " ".join(el.get("class") or [])
    except Exception:
        return False
    return bool((i and _LIKELY_CONTENT_ID_RE.search(i)) or (c and _LIKELY_CONTENT_CL_RE.search(c)))


def _score_candidate(el: Any) -> tuple[float, _CandidateFeatures]:
    """Generic scoring. ZERO gold. Returns (score, features)."""
    from bs4 import Tag
    if not isinstance(el, Tag):
        return -1.0, _CandidateFeatures(0, 0, 0, 0.0, 0, 0, 0, False, False, False, False, False)
    try:
        txt = el.get_text(" ", strip=True)
    except Exception:
        return -1.0, _CandidateFeatures(0, 0, 0, 0.0, 0, 0, 0, False, False, False, False, False)
    if not txt or len(txt) < 100:
        return -1.0, _CandidateFeatures(len(txt), 0, 0, 0.0, 0, 0, 0, False, False, False, False, False)
    links = el.find_all("a")
    link_text = 0
    for a in links:
        try:
            if a is None:
                continue
            link_text += len((a.get_text(" ", strip=True) or ""))
        except Exception:
            pass
    link_density = (link_text + 1) / (len(txt) + 1)
    headings = el.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    paras = el.find_all(["p", "div"])
    short_links = 0
    for a in links:
        try:
            if a is None:
                continue
            t = (a.get_text(" ", strip=True) or "")
            if 0 < len(t) <= 8:
                short_links += 1
        except Exception:
            pass
    chapter_hits = len(_LAW_CHAPTER_RE.findall(txt))
    article_hits = len(_LAW_ARTICLE_RE.findall(txt))
    front = txt[: max(500, len(txt) // 3)]
    has_first_chapter_front = bool(_FIRST_CHAPTER_RE.search(front))
    has_first_article_front = bool(_FIRST_ARTICLE_RE.search(front))
    start_hit = bool(has_first_chapter_front or has_first_article_front)
    end_hit = bool(_LAW_ENFORCEMENT_END_RE.search(txt)) or article_hits >= 40
    tail_match = bool(_PAGINATION_TAIL_RE.search(txt[-300:]))
    score = (
        len(txt) * 1.0
        + len(headings) * 500.0
        + len(paras) * 30.0
        - link_density * 5000.0
        - short_links * 80.0
        + chapter_hits * 2000.0
        + article_hits * 3000.0
        + end_hit * 4500.0
        + (5500.0 if start_hit else 0.0)
        + (2000.0 if has_first_article_front else 0.0)
        + (2000.0 if has_first_chapter_front else 0.0)
        - (1500.0 if tail_match else 0.0)
    )
    feat = _CandidateFeatures(
        length=len(txt),
        n_headings=len(headings),
        n_paragraphs=len(paras),
        link_density=round(link_density, 3),
        short_links=short_links,
        chapter_hits=chapter_hits,
        article_hits=article_hits,
        has_first_chapter_front=has_first_chapter_front,
        has_first_article_front=has_first_article_front,
        start_hit=start_hit,
        end_hit=end_hit,
        tail_penalty=tail_match,
    )
    return score, feat


# =============================================================================
# Core DOM → Frozen ParsedDocument
# =============================================================================
def _extract_blocks_from_root(root: Any, doc_id_for_trace: str, warnings: list[str]) -> list[Block]:
    """
    Extract semantic blocks from a main-content candidate (BeautifulSoup Tag).
    Block types: 6 frozen values only (HEADING/PARAGRAPH/LIST_ITEM/TABLE/METADATA/UNKNOWN).
    Unknown or ambiguous content is preserved as UNKNOWN blocks (never silent dropped).
    """
    from bs4 import NavigableString, Comment, Tag
    blocks: list[Block] = []
    order_counter = 0

    def _append(b: Block) -> None:
        nonlocal order_counter
        object.__setattr__(b, "order", order_counter)
        object.__setattr__(b, "block_id", Block.make_block_id(doc_id_for_trace, order_counter))
        blocks.append(b)
        order_counter += 1

    def _unknown_or_paragraph(txt: str, reason: str) -> None:
        if not txt.strip():
            return
        stripped = txt.strip()
        if len(stripped) >= 2:
            if 6 <= len(stripped) <= 4000 and (stripped[:1].isalpha() or stripped[:1] in "第本一二三四五六七八九十百零〇0123456789\"'（(【《"):
                _append(Block("tmp", BlockType.PARAGRAPH, stripped, 0))
            else:
                _append(Block("tmp", BlockType.UNKNOWN, stripped, 0, metadata={"reason": reason}))
                warnings.append(f"UNKNOWN block emitted, reason={reason}, len={len(stripped)}")

    def _make_table_data(table_el: Tag) -> Optional[TableData]:
        """Convert <table> DOM to frozen TableData (NEVER flatten first)."""
        if not isinstance(table_el, Tag):
            return None
        caption_el = table_el.find("caption")
        caption: Optional[str] = None
        if caption_el is not None:
            caption = caption_el.get_text(" ", strip=True) or None

        rows_dom: list[Any] = []
        thead = table_el.find("thead")
        tbody = table_el.find("tbody")
        if thead or tbody:
            if thead:
                rows_dom.extend(thead.find_all("tr"))
            if tbody:
                rows_dom.extend(tbody.find_all("tr"))
        else:
            rows_dom = table_el.find_all("tr")

        if not rows_dom:
            return None

        grid: list[list[TableCell]] = []
        for row_dom in rows_dom:
            if not isinstance(row_dom, Tag):
                continue
            cells_dom = row_dom.find_all(["th", "td"])
            row: list[TableCell] = []
            for cd in cells_dom:
                if not isinstance(cd, Tag):
                    continue
                try:
                    txt = cd.get_text(" ", strip=True) or ""
                except Exception:
                    txt = ""
                try:
                    cs = int(cd.get("colspan") or 1)
                except Exception:
                    cs = 1
                try:
                    rs = int(cd.get("rowspan") or 1)
                except Exception:
                    rs = 1
                is_h = (cd.name == "th") or bool(list(cd.parents) and any(p.name == "thead" for p in cd.parents))
                row.append(TableCell(text=txt, colspan=max(1, cs), rowspan=max(1, rs), is_header=bool(is_h)))
            grid.append(row)

        if grid and all(not c.is_header for c in grid[0]):
            n_cols = sum(c.colspan for c in grid[0])
            if n_cols >= 2 and len(grid) >= 2:
                for i in range(len(grid[0])):
                    object.__setattr__(grid[0][i], "is_header", True)
        if not grid:
            return None

        header_cells = list(grid[0])
        header_logical_cols = sum(c.colspan for c in header_cells)

        def _logical_len(row: list[TableCell]) -> int:
            return sum(c.colspan for c in row)

        data_rows_out: list[list[TableCell]] = []
        for raw_row in grid[1:]:
            padded = list(raw_row)
            while _logical_len(padded) < header_logical_cols:
                padded.append(TableCell(text="", colspan=1))
            trimmed: list[TableCell] = []
            logical = 0
            for c in padded:
                if logical + c.colspan > header_logical_cols:
                    needed = max(1, header_logical_cols - logical)
                    trimmed.append(TableCell(text=c.text, colspan=needed, is_header=c.is_header))
                    logical += needed
                    break
                trimmed.append(c)
                logical += c.colspan
                if logical >= header_logical_cols:
                    break
            if len(trimmed) < len(header_cells):
                while len(trimmed) < len(header_cells):
                    trimmed.append(TableCell(text=""))
            elif len(trimmed) > len(header_cells):
                merged = list(trimmed[: len(header_cells)])
                surplus = trimmed[len(header_cells):]
                if merged:
                    last = merged[-1]
                    new_text = (last.text + " " + " ".join(s.text for s in surplus if s.text)).strip()
                    merged[-1] = TableCell(text=new_text, colspan=last.colspan,
                                          rowspan=last.rowspan, is_header=last.is_header)
                trimmed = merged
            data_rows_out.append(trimmed)

        if not header_cells:
            return None
        try:
            return TableData(headers=header_cells, rows=data_rows_out, caption=caption)
        except ContractViolation:
            warnings.append("HTML <table> failed TableData Contract; emitted as UNKNOWN block instead (preserved)")
            return None

    def _traverse(elem: Any, list_level: int = 0) -> None:
        nonlocal order_counter
        if isinstance(elem, (NavigableString, Comment)):
            return
        if not isinstance(elem, Tag):
            return
        try:
            txt = elem.get_text(" ", strip=True)
        except Exception:
            txt = ""
        if not txt:
            return
        name = elem.name.lower() if isinstance(getattr(elem, "name", None), str) else "?"
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            _append(Block("tmp", BlockType.HEADING, txt, 0, level=level))
            return
        if name == "table":
            td = _make_table_data(elem)
            if td is None:
                try:
                    raw_text = elem.get_text(" ", strip=True) or ""
                except Exception:
                    raw_text = ""
                if raw_text:
                    _unknown_or_paragraph(raw_text, reason="html_table_failed_to_structuralize")
                return
            md = td.to_markdown()
            _append(Block("tmp", BlockType.TABLE, md, 0, table_data=td))
            return
        if name in ("ul", "ol"):
            for li in elem.find_all("li", recursive=False):
                try:
                    li_txt = li.get_text(" ", strip=True) or ""
                except Exception:
                    li_txt = ""
                if not li_txt:
                    continue
                _append(Block("tmp", BlockType.LIST_ITEM, li_txt, 0, metadata={"list_level": max(0, list_level)}))
            return
        if name in ("p", "div", "section", "article", "blockquote", "td", "span", "li"):
            has_child_block = False
            try:
                for child in elem.find_all(
                    ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table", "section", "article", "blockquote"],
                    recursive=False,
                ):
                    has_child_block = True
                    break
            except Exception:
                has_child_block = False
            if not has_child_block and len(txt) >= 2:
                _append(Block("tmp", BlockType.PARAGRAPH, txt, 0))
                return
        for child in elem.find_all(recursive=False):
            try:
                _traverse(child, list_level=list_level)
            except Exception as e:
                warnings.append(f"subtree_traverse crashed: {type(e).__name__}: {e}")
                try:
                    fallback = child.get_text(" ", strip=True) or ""
                except Exception:
                    fallback = ""
                if fallback:
                    _unknown_or_paragraph(fallback, reason=f"subtree_crash:{type(e).__name__}")
        # If no child tags exist (children are purely NavigableStrings / custom text nodes)
        # and elem wasn't handled by any known structural rule above — PRESERVE content.
        try:
            has_child_tags = bool(elem.find_all(recursive=False))
        except Exception:
            has_child_tags = True
        if (not has_child_tags) and len(txt) >= 2 and name not in (
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "div", "section", "article", "blockquote", "td", "span", "li",
        ):
            _unknown_or_paragraph(txt, reason=f"custom_tag:{name}")

    try:
        _traverse(root)
    except Exception as e:
        warnings.append(f"top-level block extraction crashed: {type(e).__name__}: {e}")
        try:
            fallback = root.get_text(" ", strip=True) or ""
        except Exception:
            fallback = ""
        if fallback:
            _unknown_or_paragraph(fallback, reason=f"top-level_block_extract_crash:{type(e).__name__}")
    return blocks


def _extract_html_into_blocks(raw_html: str, *, warnings: list[str]) -> tuple[str, dict[str, Any], list[Block], dict[str, Any]]:
    """Pure HTML → (title, metadata, blocks, flags). All generic, no gold."""
    from bs4 import BeautifulSoup, Tag
    parser_flags: dict[str, Any] = {"pagination_union_used": False, "flat_dom_fallback_used": False}

    soup = BeautifulSoup(raw_html, "html.parser")
    warnings.append(f"dom_backend::{PARSER_BACKEND}")

    STRIP_TAGS = ["script", "style", "noscript", "svg", "button", "form", "iframe"]
    for t in STRIP_TAGS:
        for n in soup.find_all(t):
            try:
                n.decompose()
            except Exception:
                pass
    for sem in ("nav", "footer", "header", "aside"):
        for n in soup.find_all(sem):
            try:
                has_content_marker_child = any(
                    _is_likely_content_marker(c)
                    for c in n.find_all(["div", "article", "section", "main", "td", "table"])
                )
            except Exception:
                has_content_marker_child = False
            if has_content_marker_child:
                try:
                    n.name = "div"
                    continue
                except Exception:
                    pass
            try:
                n.decompose()
            except Exception:
                pass
    for container in list(soup.find_all(["div", "ul", "ol", "section", "span"])):
        try:
            ident = (container.get("id") or "") + " " + " ".join(container.get("class") or [])
        except Exception:
            ident = ""
        if not ident or not _NAV_CLASS_ID_RE.search(ident):
            continue
        if _is_likely_content_marker(container):
            continue
        try:
            has_head = bool(container.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))
            inner = container.get_text(" ", strip=True)
            if has_head or len(inner) >= 400:
                continue
        except Exception:
            continue
        try:
            container.decompose()
        except Exception:
            pass

    best: Optional[Tag] = None
    best_score = -1e18
    scored_pool: list[tuple[float, _CandidateFeatures, Tag]] = []
    search_roots = [soup]
    if soup.body and soup.body is not soup:
        search_roots.append(soup.body)
    for root in search_roots:
        for spec in _MAIN_CONTENT_SPEC:
            try:
                if "tag" in spec:
                    nodes = root.find_all(spec["tag"])
                elif "role" in spec:
                    nodes = root.find_all(attrs={"role": spec["role"]})
                elif "class_re" in spec:
                    pat = re.compile(spec["class_re"])
                    nodes = [el for el in root.find_all(["div", "article", "section", "main", "td", "table"])
                             if any(pat.search(c or "") for c in (el.get("class") or []))]
                elif "id_re" in spec:
                    pat = re.compile(spec["id_re"])
                    nodes = [el for el in root.find_all(["div", "article", "section", "main", "td", "table"])
                             if pat.search(el.get("id") or "")]
                else:
                    nodes = []
            except Exception:
                nodes = []
            for el in nodes:
                if not isinstance(el, Tag):
                    continue
                s, fe = _score_candidate(el)
                if s < 0:
                    continue
                scored_pool.append((s, fe, el))
                if s > best_score:
                    best_score, best = s, el
    try:
        fallback_scan = soup.find_all(["div", "article", "section", "td", "table", "main"])
    except Exception:
        fallback_scan = []
    for el in fallback_scan:
        if not isinstance(el, Tag):
            continue
        try:
            t = el.get_text(" ", strip=True) or ""
        except Exception:
            t = ""
        if len(t) < 100:
            continue
        s, fe = _score_candidate(el)
        if s < 0:
            continue
        scored_pool.append((s, fe, el))
        if s > best_score:
            best_score, best = s, el

    # Explicit pagination-signal scan: short containers with strong start/end structural
    # signal are valuable union candidates. Catch them even when below size thresholds above.
    try:
        for el in soup.find_all(["div", "article", "section", "td", "table"]):
            if not isinstance(el, Tag):
                continue
            try:
                t = el.get_text(" ", strip=True) or ""
            except Exception:
                t = ""
            if len(t) < 80:
                continue
            s, fe = _score_candidate(el)
            if s < 0:
                continue
            if fe.start_hit or fe.end_hit or fe.article_hits >= 3:
                # Only add if not already a duplicate score+element
                if not any(pool[2] is el for pool in scored_pool):
                    scored_pool.append((s, fe, el))
                if s > best_score:
                    best_score, best = s, el
    except Exception:
        pass

    selected: list[Tag] = []
    if best is None:
        parser_flags["flat_dom_fallback_used"] = True
        warnings.append("flat_dom_fallback::no named candidates matched")
        fallback_candidates: list[tuple[float, Tag, str]] = []
        for el in soup.find_all(["article", "section", "div", "td"]):
            if not isinstance(el, Tag):
                continue
            try:
                t = el.get_text(" ", strip=True) or ""
            except Exception:
                t = ""
            if len(t) >= 100:
                s, _ = _score_candidate(el)
                if s >= 0:
                    fallback_candidates.append((s, el, t))
        fallback_candidates.sort(key=lambda x: -x[0])
        if fallback_candidates:
            selected = [fallback_candidates[0][1]]
    else:
        alt_thresh = best_score * 0.15
        top_sorted = sorted(scored_pool, key=lambda x: -x[0])[:12]
        # Keep candidates passing alt_threshold OR that have high signal start/end
        # (pagination page 1 of N rarely out-scores the final page in raw length, but its
        # start anchor is exactly what single-best misses).
        top_pool = []
        for rec in top_sorted:
            s_r, f_r, el_r = rec
            if s_r >= alt_thresh or f_r.start_hit or f_r.end_hit:
                top_pool.append(rec)
        deduped: list[tuple[float, _CandidateFeatures, Tag]] = []
        dominated: set[int] = set()
        for i, (s, f, el) in enumerate(top_pool):
            if i in dominated:
                continue
            keep = True
            for j, (s2, f2, el2) in enumerate(top_pool):
                if i == j or j in dominated:
                    continue
                try:
                    if el2 is not el and el in el2.descendants:
                        keep = False
                        dominated.add(i)
                        break
                except Exception:
                    pass
            if keep:
                deduped.append((s, f, el))
        if len(deduped) >= 2:
            def _has_start(txt: str) -> bool:
                front = txt[: max(500, len(txt) // 3)]
                return bool(_FIRST_CHAPTER_RE.search(front) or _FIRST_ARTICLE_RE.search(front))
            def _has_end(txt: str) -> bool:
                return bool(_LAW_ENFORCEMENT_END_RE.search(txt)) or len(_LAW_ARTICLE_RE.findall(txt)) >= 15
            try:
                best_txt = best.get_text(" ", strip=True) or ""
            except Exception:
                best_txt = ""
            s_start = _has_start(best_txt)
            s_end = _has_end(best_txt)
            union_parts: list[str] = []
            for s, f, el in deduped[:3]:
                try:
                    union_parts.append(el.get_text(" ", strip=True) or "")
                except Exception:
                    pass
            union_txt = "\n".join(union_parts)
            u_start = _has_start(union_txt)
            u_end = _has_end(union_txt)
            missing_single = (not s_start) or (not s_end)
            fills_half = ((not s_start and u_start) or (not s_end and u_end))
            complete_pair = (u_start and u_end) and not (s_start and s_end)
            best_arts = len(_LAW_ARTICLE_RE.findall(best_txt))
            union_arts = len(_LAW_ARTICLE_RE.findall(union_txt))
            article_boost = (union_arts > 1.15 * max(1, best_arts))
            if missing_single or fills_half or complete_pair or article_boost:
                # Build chosen list: always include the best, plus any complementary candidates
                # that provide the missing start or end anchor.
                chosen = [deduped[0]]
                included_ids = {id(deduped[0][2])}
                best_start = False
                best_end = False
                try:
                    best_t = deduped[0][2].get_text(" ", strip=True) or ""
                except Exception:
                    best_t = ""
                def _hs(t):
                    front = t[: max(500, len(t) // 3)]
                    return bool(re.search(r"第一章|第1章|总则", front) or re.search(r"第一条|第1条", front))
                def _he(t):
                    return bool(_LAW_ENFORCEMENT_END_RE.search(t) or len(_LAW_ARTICLE_RE.findall(t)) >= 40)
                best_start = _hs(best_t)
                best_end = _he(best_t)
                for s, f, el in deduped[1:]:
                    if id(el) in included_ids:
                        continue
                    try:
                        tx = el.get_text(" ", strip=True) or ""
                    except Exception:
                        tx = ""
                    cstart = _hs(tx)
                    cend = _he(tx)
                    need = False
                    if (not best_start) and cstart:
                        need = True
                    if (not best_end) and cend:
                        need = True
                    if need:
                        chosen.append((s, f, el))
                        included_ids.add(id(el))
                        best_start = best_start or cstart
                        best_end = best_end or cend
                    if len(chosen) >= 3:
                        break
                def _start_rank(rec: tuple[float, _CandidateFeatures, Tag]) -> int:
                    _, _, t = rec
                    try:
                        tx = t.get_text(" ", strip=True) or ""
                    except Exception:
                        tx = ""
                    return 1 if _has_start(tx) else 0
                try:
                    chosen = sorted(chosen, key=lambda r: -_start_rank(r))
                except Exception:
                    pass
                selected = [el for (_, _, el) in chosen]
                parser_flags["pagination_union_used"] = True
                warnings.append(
                    f"pagination_union::selected {len(selected)} disjoint containers "
                    f"(single start={s_start},end={s_end}; union start={u_start},end={u_end})"
                )
        if not selected:
            selected = [best]

    # Safety net: if selected still empty (tiny HTML / no candidates pass threshold) → body/soup
    if not selected:
        parser_flags["flat_dom_fallback_used"] = True
        warnings.append("flat_dom_fallback::no candidates; falling back to root body/soup")
        try:
            if soup.body and soup.body is not soup:
                selected = [soup.body]
            else:
                selected = [soup]
        except Exception:
            selected = [soup]

    extracted_title: str = ""
    try:
        if soup.title and soup.title.string:
            raw_title = soup.title.get_text(" ", strip=True) or ""
            cleaned = _SITE_SUFFIX_RE.sub("", raw_title).strip()
            if 4 <= len(cleaned) <= 160:
                extracted_title = cleaned
    except Exception:
        extracted_title = ""
    if not extracted_title:
        for elem in selected:
            try:
                h1s = elem.find_all("h1")
            except Exception:
                h1s = []
            for h in h1s:
                try:
                    t = h.get_text(" ", strip=True) or ""
                except Exception:
                    t = ""
                if len(t) >= 6:
                    extracted_title = t
                    break
            if extracted_title:
                break
    if not extracted_title:
        try:
            soup_front = (soup.get_text(" ", strip=True) or "")[:8000]
            m = _GENERIC_LEGAL_TITLE_RE.search(soup_front)
            if m:
                extracted_title = m.group(0)
        except Exception:
            extracted_title = ""

    metadata: dict[str, Any] = {}
    try:
        all_meta = soup.find_all("meta")
        for mt in all_meta:
            key = ""
            try:
                key = ((mt.get("name") or "").strip().lower()
                       or (mt.get("property") or "").strip().lower()
                       or (mt.get("itemprop") or "").strip().lower())
            except Exception:
                key = ""
            try:
                val = (mt.get("content") or "").strip()
            except Exception:
                val = ""
            if not key or not val:
                continue
            for canonical, candidates in _META_TAG_CANDIDATES.items():
                if key in candidates or any(c in key for c in candidates):
                    if canonical not in metadata or len(str(metadata[canonical])) < len(val):
                        metadata[canonical] = val
        if "publish_date" not in metadata:
            for t in soup.find_all("time"):
                try:
                    dt = t.get("datetime") or t.get_text(" ", strip=True) or ""
                except Exception:
                    dt = ""
                if dt and re.search(r"\d{4}", dt):
                    metadata["publish_date"] = dt
                    break
    except Exception as e:
        warnings.append(f"metadata_extract_warn::{type(e).__name__}:{e}")

    blocks: list[Block] = []
    order_next = [0]

    def _prep_traced_blocks(root: Tag) -> None:
        nonlocal blocks
        inner = _extract_blocks_from_root(root, "__placeholder__", warnings)
        for b in inner:
            object.__setattr__(b, "order", order_next[0])
            object.__setattr__(b, "block_id", Block.make_block_id("__placeholder__", order_next[0]))
            blocks.append(b)
            order_next[0] += 1

    for main_el in selected:
        try:
            if isinstance(main_el, Tag):
                _prep_traced_blocks(main_el)
        except Exception as e:
            warnings.append(f"candidate_block_extract_crash::{type(e).__name__}:{e}")

    if extracted_title and len(extracted_title.strip()) >= 4:
        t_norm = extracted_title.strip()
        already_present = False
        for b in blocks[:3]:
            if b.type is BlockType.HEADING and t_norm in b.text:
                already_present = True
                break
        if not already_present:
            h1 = Block("tmp", BlockType.HEADING, t_norm, 0, level=1)
            object.__setattr__(h1, "order", 0)
            object.__setattr__(h1, "block_id", Block.make_block_id("__placeholder__", 0))
            for b in blocks:
                object.__setattr__(b, "order", b.order + 1)
                object.__setattr__(b, "block_id", Block.make_block_id("__placeholder__", b.order))
            blocks.insert(0, h1)

    return extracted_title.strip(), metadata, blocks, parser_flags


# =============================================================================
# PUBLIC PRODUCTION-LIKE API
# =============================================================================
def _read_raw_html(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as e:
        raise ValueError(f"HTML source file does not exist: {path}") from e


def parse_html_bytes(
    raw_html: bytes | str,
    *,
    source_file: str,
    document_id: str,
) -> ParsedDocument:
    if not document_id or not isinstance(document_id, str):
        raise ValueError("document_id must be a non-empty string (HTML Parser V2 contract)")
    if not source_file or not isinstance(source_file, str):
        raise ValueError("source_file must be a non-empty string (HTML Parser V2 contract)")
    if isinstance(raw_html, bytes):
        try:
            html_str = raw_html.decode("utf-8", errors="replace")
        except Exception:
            html_str = raw_html.decode("latin-1", errors="replace")
    elif isinstance(raw_html, str):
        html_str = raw_html
    else:
        raise ValueError("raw_html must be bytes or str")
    if not html_str.strip():
        raise ValueError("Empty HTML document; no content to parse")

    warnings: list[str] = [f"dom_api::{PARSER_DOM_API}"]
    flags: dict[str, Any] = {}
    extracted_title = ""
    metadata: dict[str, Any] = {}
    blocks: list[Block] = []
    try:
        extracted_title, metadata, blocks, flags = _extract_html_into_blocks(html_str, warnings=warnings)
    except Exception as e:
        warnings.append(f"parser_full_crash::{type(e).__name__}:{e}")
        fallback_text = re.sub(r"<[^>]+>", " ", html_str)
        fallback_text = re.sub(r"\s+", " ", fallback_text).strip()
        blocks = [Block("tmp", BlockType.UNKNOWN, fallback_text, 0,
                        metadata={"reason": f"parser_crash_recovery:{type(e).__name__}"})]
        flags = {"parser_crashed": True}
    if not blocks:
        warnings.append("zero_blocks_emitted::UNKNOWN placeholder preserved")
        blocks = [Block("tmp", BlockType.UNKNOWN, "", 0, metadata={"reason": "zero_blocks"})]
    if flags.get("flat_dom_fallback_used"):
        warnings.append("parser_flag::flat_dom_fallback_used")
    if flags.get("pagination_union_used"):
        warnings.append("parser_flag::pagination_union_used")
    if not extracted_title:
        warnings.append("title_extraction::failed_to_extract_title_from_dom")
    if extracted_title and "title" not in metadata:
        metadata["title"] = extracted_title

    ParsedDocument.prepare_blocks(document_id, blocks)
    for b in blocks:
        b.attach_provenance(source_file)
    doc = ParsedDocument(
        document_id=document_id,
        source_file=source_file,
        title=extracted_title,
        metadata=dict(metadata),
        blocks=blocks,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        warnings=list(warnings),
    )
    doc.validate()
    return doc


def parse_html(
    path: "os.PathLike[str] | str | Path",
    *,
    document_id: Optional[str] = None,
) -> ParsedDocument:
    p = Path(path)
    if document_id is None:
        document_id = p.stem
    raw = _read_raw_html(p)
    return parse_html_bytes(raw, source_file=str(p), document_id=document_id)


__all__ = [
    "PARSER_NAME",
    "PARSER_VERSION",
    "PARSER_DOM_API",
    "PARSER_BACKEND",
    "PARSER_NEW_DEPENDENCIES",
    "LegacyAdapter",
    "LegacyParsedDocument",
    "Block",
    "BlockType",
    "TableCell",
    "TableData",
    "Provenance",
    "ParsedDocument",
    "ContractViolation",
    "parse_html",
    "parse_html_bytes",
]
