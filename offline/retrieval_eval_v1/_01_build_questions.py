"""
Task 1: Build retrieval_questions.jsonl (≥100 queries: 50 A类 contract fact + 50 B类 legal clause).
A类: 10 SYN contracts × 5 queries each (覆盖 10 大维度: amount, parties, dates, auto_renewal, payment,
     breach, termination, governing_law, product_qty, tax_rate)
B类: 20 anchor from legal_eval_v1.jsonl (LQ-001..020) + 30 extra generated from LAW docs via canonical chunks
     using heading/article/sub章 → answerable query.
All queries must NOT fabricate GT values — bind them strictly to files.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from _reval_utils import (  # noqa: E402
    CANON_B_CHUNKS_PATH,
    LEGAL_EVAL_PATH,
    QUESTIONS_PATH,
    SEED,
    SYN_FACTS_PATH,
    ensure_dirs,
    read_json,
    read_jsonl,
    set_global_seed,
    write_jsonl,
)

ensure_dirs()
set_global_seed(SEED)


# ======================================================================
# A 类: Contract fact queries (50 条)
# ======================================================================

# 10 大维度 × 每份合同选择 5 个维度 (保证整体 10 维度每项 ≥ 5 queries)
DIMENSIONS = [
    "contract_amount",   # A-1: easy
    "party_info",        # A-2: easy (party_a, party_b, 交替)
    "sign_dates",        # A-3: easy-medium (sign/effective/expiry)
    "auto_renewal",      # A-4: medium
    "payment_terms",     # A-5: medium
    "breach_terms",      # A-6: medium-hard
    "termination_notice",# A-7: medium
    "governing_law",     # A-8: hard
    "product_qty",       # A-9: hard (important_clause/products_top)
    "tax_rate",          # A-10: easy-medium
]

# 每份合同选择 5 个维度: 轮询分配，保证 10 维度 × (10 合同 × 5/10)= 每维 5
SYN_DIMENSION_ALLOCATION = [
    # SYN-001..010; each takes 5 distinct dims
    [0, 1, 2, 5, 9],          # 001: amount, parties, sign, breach, tax
    [0, 3, 4, 7, 9],          # 002: amount, auto_renew, payment, governing, tax
    [1, 2, 6, 7, 8],          # 003: parties, sign, termination, governing, product
    [0, 3, 5, 6, 8],          # 004: amount, auto_renew, breach, termination, product
    [1, 2, 4, 7, 9],          # 005: parties, sign, payment, governing, tax
    [0, 2, 5, 7, 8],          # 006: amount, sign, breach, governing, product
    [1, 3, 4, 6, 9],          # 007: parties, auto_renew, payment, termination, tax
    [0, 3, 5, 7, 8],          # 008: amount, auto_renew, breach, governing, product
    [1, 2, 4, 6, 9],          # 009: parties, sign, payment, termination, tax
    [0, 2, 3, 5, 8],          # 010: amount, sign, auto_renew, breach, product
]


def _fmt_amount(v: int | None, currency: str = "CNY") -> list[str]:
    if v is None:
        return []
    # 支持多种数字变体（可能文本写 "1,200,000元" / 中文 / 纯数字）
    vs = []
    vs.append(f"{v:,}元")
    vs.append(f"{v:,}")
    vs.append(str(v))
    # 大写金额省略（文本里几乎不写）
    return vs


def _value_substrings(value) -> list[str]:
    """Turn a GT value into candidate substring list for evidence match."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        if isinstance(value, float) and abs(value - int(value)) < 1e-9:
            return _fmt_amount(int(value))
        # tax rate float -> percent
        pct = f"{value*100:g}%"
        return [pct, f"{value}"]
    if isinstance(value, bool):
        return ["自动续约" if value else "不自动续约", "自动续期" if value else "不自动续期"]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        out = [s]
        # if it looks like a date YYYY-MM-DD also add YYYY年M月D日 variants
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
        if m:
            y, mo, d = m.groups()
            out.append(f"{y}年{int(mo)}月{int(d)}日")
            out.append(f"{y}-{mo}-{d}")
        return out
    if isinstance(value, dict):
        name = value.get("name")
        if name:
            return [name, value.get("short", ""), value.get("uscc", "")]
        return [str(v) for v in value.values() if isinstance(v, str) and v]
    return [str(value)]


def _build_contract_query(syn: dict, dim_idx: int, qseq: int) -> dict | None:
    sid = syn["synthetic_contract_id"]
    syn_idx = int(sid.split("-")[1]) - 1
    dim = DIMENSIONS[dim_idx]
    qid = f"CQ-{qseq:03d}"

    gt_fact: dict = {"target_field": dim}
    ev_values: list[str] = []
    query = ""
    difficulty = "easy"
    ct = syn.get("contract_type", "")

    # 从 facts 中提取 + 构造问法 (variations based on syn_idx)
    variations_by_dim = {
        "contract_amount": [
            f"{syn['party_a']['name']}与{syn['party_b']['name']}签订的合同总金额是多少？",
            f"该合同总价为多少人民币？合同款额是多少？",
            f"{syn['contract_number']}号合同的总金额（含税）是多少元？",
            f"根据{suffix_ct(ct)}合同，标的款项合计是多少？",
            f"请查询{suffix_ct(ct)}合同约定的合同金额。",
        ],
        "party_info": [
            f"{syn['contract_number']}合同的甲方（委托方）是谁？",
            f"合同编号{syn['contract_number']}中，乙方名称是？",
            f"本{suffix_ct(ct)}合同的甲乙方公司全称分别是？",
            f"{syn['sign_date']}签订的{suffix_ct(ct)}合同，乙方为哪家公司？",
            f"请说出{suffix_ct(ct)}合同双方的企业名称。",
        ],
        "sign_dates": [
            f"{syn['contract_number']}号合同的签订日期是哪一天？",
            f"该{suffix_ct(ct)}合同的生效日期和到期日期分别是什么时候？",
            f"合同何时到期？到期日是？",
            f"请查询{suffix_ct(ct)}合同开始生效的时间。",
            f"{syn['party_a']['name']}签订的{suffix_ct(ct)}合同履行到哪一天结束？",
        ],
        "auto_renewal": [
            f"合同到期后是否自动续约？续约条件是什么？",
            f"{syn['contract_number']}合同约定的续约条款是怎样的？是否自动续期？",
            f"该{suffix_ct(ct)}合同期满后是否自动延长？延长期限是多少？",
            f"请问{suffix_ct(ct)}合同期满是否会自动续签？若续约，续多久？",
            f"是否自动续约条款查询。",
        ],
        "payment_terms": [
            f"{syn['contract_number']}合同约定的付款方式是什么？",
            f"甲方向乙方支付{suffix_ct(ct)}合同款项的付款条件和时间节点？",
            f"请说明{suffix_ct(ct)}合同项下的付款安排（付款条款摘要）。",
            f"{syn['party_b']['name']}向甲方开具何种发票？款项分几次支付？",
            f"{suffix_ct(ct)}合同的付款比例和付款节点是？",
        ],
        "breach_terms": [
            f"乙方逾期交付{suffix_ct(ct)}合同标的物的违约金率是多少？上限？",
            f"{syn['contract_number']}合同对违约方的违约责任有什么约定？",
            f"{suffix_ct(ct)}合同违约金上限占合同金额的百分比是？",
            f"甲方逾期付款需要支付多少滞纳金/违约金？",
            f"请查询{suffix_ct(ct)}合同中关于违约和赔偿责任的条款。",
        ],
        "termination_notice": [
            f"解除{suffix_ct(ct)}合同需要提前多少天书面通知？",
            f"{syn['contract_number']}合同提前解除的通知期限是多少日？",
            f"任何一方要求不再续约或提前解除需提前多少天通知？",
            f"{suffix_ct(ct)}合同终止条款中，解约提前通知期多长？",
            f"提前终止合同的期限：解除{suffix_ct(ct)}合同时守约方应提前几日通知？",
        ],
        "governing_law": [
            f"{syn['contract_number']}合同项下发生争议，由哪个法院或仲裁委管辖？",
            f"{suffix_ct(ct)}合同的争议解决机构是？管辖法院？",
            f"本{suffix_ct(ct)}合同约定的纠纷解决方式是诉讼还是仲裁？具体机构？",
            f"因{suffix_ct(ct)}合同引发的争议，应到哪里起诉或申请仲裁？",
            f"请查询{suffix_ct(ct)}合同的管辖法院/仲裁条款。",
        ],
        "product_qty": [
            f"{suffix_ct(ct)}合同项下的产品/服务数量是多少？（吨/千克/件）",
            f"{syn['contract_number']}合同约定的标的数量规模？（重量或件数）",
            f"{suffix_ct(ct)}合同重要条款中关于标的物数量与比例的说明？",
            f"请说明{suffix_ct(ct)}合同下的产品数量、佣金比例、保底等数量信息。",
            f"{suffix_ct(ct)}合同重要事实中的量化指标有哪些（百分比/吨/保底金额等）？",
        ],
        "tax_rate": [
            f"{syn['contract_number']}合同的增值税税率是百分之几？",
            f"{suffix_ct(ct)}合同约定的税率是多少？乙方开具发票税率？",
            f"请查询{suffix_ct(ct)}合同含税的适用税率（百分比）。",
            f"{syn['party_b']['name']}开具增值税专用发票时，{suffix_ct(ct)}合同税率为多少？",
            f"{suffix_ct(ct)}合同重要条款中载明的税率、发票类型分别是？",
        ],
    }

    q_list = variations_by_dim[dim]
    query = q_list[syn_idx % len(q_list)]

    # Extract GT
    if dim == "contract_amount":
        v = syn.get("amount")
        ev_values += _fmt_amount(int(v)) if v else []
        ev_values.append(syn.get("currency", ""))
        gt_fact["amount"] = v
        gt_fact["currency"] = syn.get("currency")
        difficulty = "easy"
    elif dim == "party_info":
        pa = syn["party_a"]; pb = syn["party_b"]
        # 交替问甲方 or 乙方 or 双方
        variant = syn_idx % 3
        if variant == 0:
            ev_values += [pa["name"], pa.get("short", ""), pa.get("uscc", "")]
            gt_fact["party_a"] = pa["name"]
            gt_fact["target_subfield"] = "party_a"
        elif variant == 1:
            ev_values += [pb["name"], pb.get("short", ""), pb.get("uscc", "")]
            gt_fact["party_b"] = pb["name"]
            gt_fact["target_subfield"] = "party_b"
        else:
            ev_values += [pa["name"], pb["name"], pa.get("short", ""), pb.get("short", "")]
            gt_fact["party_a"] = pa["name"]
            gt_fact["party_b"] = pb["name"]
            gt_fact["target_subfield"] = "both"
        difficulty = "easy"
    elif dim == "sign_dates":
        sd = syn.get("sign_date"); ed = syn.get("effective_date"); expd = syn.get("expiry_date")
        variant = syn_idx % 3
        if variant == 0:
            ev_values += _value_substrings(sd)
            gt_fact["sign_date"] = sd
            gt_fact["target_subfield"] = "sign_date"
            difficulty = "easy"
        elif variant == 1:
            ev_values += _value_substrings(ed)
            gt_fact["effective_date"] = ed
            gt_fact["target_subfield"] = "effective_date"
            difficulty = "easy"
        else:
            ev_values += _value_substrings(expd)
            gt_fact["expiry_date"] = expd
            gt_fact["target_subfield"] = "expiry_date"
            difficulty = "medium"
    elif dim == "auto_renewal":
        ar = syn.get("auto_renewal")
        rp = syn.get("renewal_period")
        ev_values += _value_substrings(ar)
        if rp:
            ev_values.append(rp)
        gt_fact["auto_renewal"] = ar
        gt_fact["renewal_period"] = rp
        difficulty = "medium"
    elif dim == "payment_terms":
        pt = syn.get("payment_terms", "")
        # 提取关键子串：金额数字 + 百分比 + 节点词
        ev_values += re.findall(r"\d[\d,]*\.?\d*(?:%|元)?", pt)[:8]
        ev_values.append(syn.get("contract_number", ""))
        gt_fact["payment_terms"] = pt
        difficulty = "medium"
    elif dim == "breach_terms":
        bt = syn.get("breach_terms", "")
        ev_values += re.findall(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?‰", bt)
        # 关键词
        for kw in ["违约金", "上限", "赔偿", "滞纳金", "泄露"]:
            if kw in bt:
                ev_values.append(kw)
        gt_fact["breach_terms"] = bt
        difficulty = "hard"
    elif dim == "termination_notice":
        tnd = syn.get("termination_notice_days")
        if tnd:
            ev_values += [f"{tnd}日", f"{tnd}天", f"提前{tnd}日", f"{tnd}"]
        tt = syn.get("termination_terms", "")
        for kw in ["解除", "书面通知", "终止", "重大违约"]:
            if kw in tt:
                ev_values.append(kw)
        gt_fact["termination_notice_days"] = tnd
        gt_fact["termination_terms"] = tt
        difficulty = "medium"
    elif dim == "governing_law":
        gdr = syn.get("governing_or_dispute_resolution", "")
        # 提取机构名 + "仲裁"/"法院"
        for m in re.finditer(r"(?:上海仲裁委员会|南京仲裁委员会|中国国际经济贸易仲裁委员会|北京市.*?法院|上海市.*?法院|杭州市.*?法院|深圳市.*?法院|苏州市.*?法院|南京市.*?法院|人民法院|仲裁委员会)", gdr):
            ev_values.append(m.group(0))
        if not ev_values:
            ev_values.append(gdr[:40])
        gt_fact["governing_or_dispute_resolution"] = gdr
        difficulty = "hard"
    elif dim == "product_qty":
        icf = syn.get("important_clause_facts", {})
        # 提取百分比 + 吨/千克/金额/保底 等量化指标
        qty_kw = []
        for v in icf.values():
            if isinstance(v, (int, float)):
                qty_kw.extend(_value_substrings(v))
            elif isinstance(v, str):
                qty_kw.extend(re.findall(r"\d[\d,]*\.?\d*(?:%|吨|千克|kg|公斤|元|万)", v))
        if not qty_kw:
            # 从 payment_terms / breach_terms / expected_entities 兜底
            for e in syn.get("expected_entities", [])[:5]:
                qty_kw.append(e)
        ev_values += qty_kw[:10]
        gt_fact["important_clause_facts"] = icf
        difficulty = "hard"
    elif dim == "tax_rate":
        tr = syn.get("important_clause_facts", {}).get("tax_rate")
        if tr is not None:
            ev_values += _value_substrings(tr)
        ev_values.append("增值税专用发票")
        gt_fact["tax_rate"] = tr
        difficulty = "easy-medium" if False else "medium"

    # Filter out empty ev values
    ev_values = [v for v in ev_values if isinstance(v, str) and len(v) > 0]
    # Deduplicate while preserving order
    seen = set(); ev_values = [x for x in ev_values if not (x in seen or seen.add(x))]
    if not ev_values:
        # 兜底: 合同编号
        ev_values.append(syn.get("contract_number", ""))

    return {
        "query_id": qid,
        "query": query,
        "query_type": "contract_fact",
        "difficulty": _map_diff(difficulty),
        "ground_truth_contract_id": sid,
        "ground_truth_logical_document_id": None,
        "ground_truth_fact": gt_fact,
        "evidence_match_rules": {
            "kind": "substring",
            "values": ev_values,
            "synthetic_doc_ids": [syn["file_name"].replace(".docx", "")],
            "source_template_id": syn.get("source_template_id"),
        },
        "source_template_id": syn.get("source_template_id"),
        "contract_type": ct,
        "law_article_refs": None,
        "created_by": "v1_gold_recipe",
    }


def _map_diff(x: str) -> str:
    if x in ("easy",):
        return "easy"
    if x in ("medium", "easy-medium"):
        return "medium"
    return "hard"


def suffix_ct(ct: str) -> str:
    mapping = {
        "DATA_ENTRUSTED_PROCESSING": "数据委托处理",
        "DATA_SERVICE_AGENCY": "数据中介服务",
        "PROCUREMENT_MATERIAL": "材料采购",
        "AGRICULTURAL_SALE": "农产品买卖",
        "LEASE": "租赁",
        "EPC": "工程总承包（EPC）",
        "ENTRUSTMENT": "委托",
    }
    return mapping.get(ct, "本")


def build_contract_questions(syn_facts: list[dict]) -> list[dict]:
    out: list[dict] = []
    qseq = 1
    assert len(syn_facts) == 10, f"expected 10 syn contracts, got {len(syn_facts)}"
    for i, syn in enumerate(syn_facts):
        alloc = SYN_DIMENSION_ALLOCATION[i]
        assert len(alloc) == 5, f"contract {syn['synthetic_contract_id']}: need 5 dims"
        for dim_idx in alloc:
            rec = _build_contract_query(syn, dim_idx, qseq)
            if rec:
                out.append(rec)
                qseq += 1
    # Pad if short (shouldn't happen with 10×5=50)
    while len(out) < 50:
        out.append(_build_contract_query(syn_facts[len(out) % 10], (len(out)*3) % 10, qseq))
        qseq += 1
    return out[:50]


# ======================================================================
# B 类: Legal clause queries (50 条)
# ======================================================================

def _law_doc_id_map() -> dict[str, str]:
    """Provide human-friendly name mapping for LAW_* IDs."""
    return {
        "LAW_001": "个人信息保护法",
        "LAW_007": "民法典",
        "LAW_008": "公司法",
        "LAW_009": "劳动合同法",
        "LAW_010": "电子商务法",
        "LAW_011": "数据安全法",
        "LAW_012": "网络安全法",
        "LAW_013": "仲裁法",
        "LAW_014": "民事诉讼法",
        "LAW_015": "消费者权益保护法",
    }


def build_legal_anchor_from_eval(legal_eval_rows: list[dict]) -> list[dict]:
    """Use 20 anchor queries from legal_eval_v1.jsonl (LQ-001..020)."""
    out: list[dict] = []
    # pick first 20 answerable ones
    picked = [r for r in legal_eval_rows if r.get("answerable", True)][:20]
    for i, r in enumerate(picked, start=1):
        doc_ids = r.get("expected_logical_document_ids") or []
        did = doc_ids[0] if doc_ids else "LAW_008"
        vk = r.get("verify_keywords", {})
        keywords_for_doc = vk.get(did, "") if isinstance(vk, dict) else ""
        ev_vals = []
        if isinstance(keywords_for_doc, str):
            ev_vals.append(keywords_for_doc)
        elif isinstance(keywords_for_doc, list):
            ev_vals = [str(x) for x in keywords_for_doc]
        # Add ground_truth words
        gt = r.get("ground_truth", "")
        # Extract法条 numbers
        arts = re.findall(r"第[一二三四五六七八九十百千万0-9〇两]+条", gt)
        ev_vals.extend(arts)
        ev_vals = [v for v in ev_vals if v]
        # Difficulty
        qt = r.get("question_type", "fact")
        diff = "easy" if qt == "fact" else ("medium" if qt == "scenario" else "hard")
        # law_article_refs
        refs = []
        law_name = _law_doc_id_map().get(did, did)
        for a in arts[:2]:
            refs.append(f"{law_name} {a}")
        out.append({
            "query_id": f"LQ-{i:03d}",
            "query": r["query"],
            "query_type": "legal_clause",
            "difficulty": diff,
            "ground_truth_contract_id": None,
            "ground_truth_logical_document_id": did,
            "ground_truth_fact": {
                "gt_answer_text": gt,
                "verify_keywords": keywords_for_doc,
                "chapter": None,
                "article": arts[0] if arts else None,
            },
            "evidence_match_rules": {
                "kind": "substring_and_docid",
                "values": ev_vals,
                "document_id": did,
            },
            "source_template_id": None,
            "contract_type": None,
            "law_article_refs": refs,
            "created_by": "v1_gold_recipe",
            "_anchor_source_eval_id": r.get("eval_id"),
        })
    return out


def build_legal_extra_from_canon_chunks(canon_b_chunks: list[dict], anchor_count: int, need_total: int = 50) -> list[dict]:
    """从 canonical B chunks 中抽取法律文档 heading/article，构造 30+ 条新增 query (LQ-021..050+)。
    只从 LAW_001/007/008/009/011/012/013/015 取。"""
    target_docs_quota = {
        "LAW_001": 10,  # 个保法 (总至少 8 + 这里分配 10, 含 anchor 合计足够)
        "LAW_007": 14,  # 民法典
        "LAW_008": 10,  # 公司法
        "LAW_009": 6,   # 劳动合同法
        "LAW_011": 3,   # 数据安全法
        "LAW_012": 3,   # 网络安全法
        "LAW_013": 2,   # 仲裁法
        "LAW_015": 2,   # 消费者权益保护法
    }  # 合计 10+14+10+6+3+3+2+2 = 50

    # collect doc_id → chunk records
    law_map: dict[str, list[dict]] = {}
    for ch in canon_b_chunks:
        meta = ch.get("chunk_metadata") or {}
        did = meta.get("document_id") or (meta.get("identity") or {}).get("document_id")
        if not did or not did.startswith("LAW_"):
            continue
        if did not in target_docs_quota:
            continue
        # Only use chunks with heading/article structure keywords
        txt = ch.get("text") or ""
        if re.search(r"第[\s零一二三四五六七八九十百千万0-9〇两]+条|第[\s零一二三四五六七八九十百千万0-9〇两]+[章节编]", txt):
            law_map.setdefault(did, []).append(ch)

    law_name = _law_doc_id_map()
    out: list[dict] = []
    qseq = anchor_count + 1

    ART_RE = re.compile(r"(第[\s零一二三四五六七八九十百千万0-9〇两]+条)[\s、.]?\s*([^\n。；]{0,60})")
    CHAP_RE = re.compile(r"(第[\s零一二三四五六七八九十百千万0-9〇两]+[章节])\s*([^\n。；]{0,60})")

    # 问法模板池（每条 chunk 产生 1 条 query）
    Q_TEMPLATES_ARTICLE = [
        "{law_name}{art_no}的主要内容是什么？",
        "根据{law_name}{art_no}，{topic}有什么规定？",
        "{law_name}{art_no}规定的{topic}条件或要求是什么？",
        "请阐述{law_name}{art_no}中关于{topic}的条款。",
        "《{law_name}》{art_no}中提到的{topic}，法律后果如何？",
    ]
    Q_TEMPLATES_CHAPTER = [
        "{law_name}{chap}主要涉及什么内容？",
        "{law_name}{chap}中{topic}相关的规定有哪些？",
        "《{law_name}》{chap}的主题范围是什么？",
    ]

    for did, quota in target_docs_quota.items():
        chunks = law_map.get(did, [])
        generated_for_doc = 0
        used_queries_doc = set()
        for ch in chunks:
            if generated_for_doc >= quota:
                break
            txt = ch.get("text") or ""
            # Prefer ART match
            matches = list(ART_RE.finditer(txt))
            chap_matches = list(CHAP_RE.finditer(txt))
            if matches:
                m = matches[0]
                art_no = re.sub(r"\s+", "", m.group(1))
                topic = (m.group(2) or "").strip()
                topic = topic[:25] if topic else "本条内容"
                # Select template by deterministic hash
                t_idx = (hash(art_no + str(generated_for_doc)) & 0x7fffffff) % len(Q_TEMPLATES_ARTICLE)
                qt = Q_TEMPLATES_ARTICLE[t_idx]
                query = qt.format(law_name=law_name.get(did, did), art_no=art_no, topic=topic)
                if query in used_queries_doc:
                    continue
                used_queries_doc.add(query)
                # ev values: art_no + topic snippet + first 10 non-space chars of topic
                ev_vals = [art_no, topic]
                # Add first keyword phrase
                snippet = txt[:80]
                ev_vals.append(snippet.replace("\n", " ")[:80])
                gt_fact = {
                    "chapter": None,
                    "article": art_no,
                    "topic": topic,
                    "expected_keywords": [art_no, topic],
                    "snippet_80": snippet,
                }
                refs = [f"{law_name.get(did, did)} {art_no}"]
            elif chap_matches:
                m = chap_matches[0]
                chap_title = re.sub(r"\s+", "", m.group(1))
                topic = (m.group(2) or "").strip()[:25] or "章节内容"
                t_idx = (hash(chap_title + str(generated_for_doc)) & 0x7fffffff) % len(Q_TEMPLATES_CHAPTER)
                qt = Q_TEMPLATES_CHAPTER[t_idx]
                query = qt.format(law_name=law_name.get(did, did), chap=chap_title, topic=topic)
                if query in used_queries_doc:
                    continue
                used_queries_doc.add(query)
                ev_vals = [chap_title, topic, txt[:80].replace("\n", " ")]
                gt_fact = {
                    "chapter": chap_title,
                    "article": None,
                    "topic": topic,
                    "expected_keywords": [chap_title, topic],
                    "snippet_80": txt[:80],
                }
                refs = [f"{law_name.get(did, did)} {chap_title}"]
            else:
                continue
            out.append({
                "query_id": f"LQ-{qseq:03d}",
                "query": query,
                "query_type": "legal_clause",
                "difficulty": "medium",
                "ground_truth_contract_id": None,
                "ground_truth_logical_document_id": did,
                "ground_truth_fact": gt_fact,
                "evidence_match_rules": {
                    "kind": "substring_and_docid",
                    "values": [v for v in ev_vals if v],
                    "document_id": did,
                },
                "source_template_id": None,
                "contract_type": None,
                "law_article_refs": refs,
                "created_by": "v1_gold_recipe",
            })
            qseq += 1
            generated_for_doc += 1

        # 如果 quota 不够，用 fallback: 从 chunk.text 随机挑 keyword（兜底）
        fallback_i = 0
        while generated_for_doc < quota and chunks:
            ch = chunks[fallback_i % len(chunks)]
            fallback_i += 1
            txt = ch.get("text") or ""
            sent = re.split(r"[。；\n]", txt)[0][:60]
            if not sent:
                continue
            query = f"《{law_name.get(did, did)}》中关于「{sent[:20]}」是如何规定的？"
            if query in used_queries_doc:
                continue
            used_queries_doc.add(query)
            ev_vals = [sent]
            out.append({
                "query_id": f"LQ-{qseq:03d}",
                "query": query,
                "query_type": "legal_clause",
                "difficulty": "hard",
                "ground_truth_contract_id": None,
                "ground_truth_logical_document_id": did,
                "ground_truth_fact": {
                    "chapter": None, "article": None,
                    "topic": sent,
                    "expected_keywords": [sent],
                    "snippet_80": txt[:80],
                },
                "evidence_match_rules": {"kind": "substring_and_docid", "values": ev_vals, "document_id": did},
                "source_template_id": None,
                "contract_type": None,
                "law_article_refs": [law_name.get(did, did)],
                "created_by": "v1_gold_recipe",
            })
            qseq += 1
            generated_for_doc += 1

    return out


# ======================================================================
# MAIN
# ======================================================================

def main() -> dict:
    syn_facts = read_jsonl(SYN_FACTS_PATH)
    legal_eval = read_jsonl(LEGAL_EVAL_PATH)
    canon_b_raw = read_json(CANON_B_CHUNKS_PATH)
    canon_b_chunks = canon_b_raw.get("chunks", [])

    a_q = build_contract_questions(syn_facts)
    b_anchors = build_legal_anchor_from_eval(legal_eval)
    b_extra = build_legal_extra_from_canon_chunks(canon_b_chunks, anchor_count=len(b_anchors), need_total=50)
    b_q = (b_anchors + b_extra)[:50]

    assert len(a_q) == 50, f"A类 query 数量应为 50，实际 {len(a_q)}"
    assert len(b_q) == 50, f"B类 query 数量应为 50，实际 {len(b_q)}"

    all_q = a_q + b_q
    write_jsonl(QUESTIONS_PATH, all_q)

    # 分布摘要
    dim_counter: Counter = Counter()
    for q in a_q:
        dim_counter[q["ground_truth_fact"].get("target_field", "?")] += 1
    law_counter: Counter = Counter()
    for q in b_q:
        law_counter[q["ground_truth_logical_document_id"] or "?"] += 1
    diff_counter: Counter = Counter(q["difficulty"] for q in all_q)
    summary = {
        "total": len(all_q),
        "contract_fact": len(a_q),
        "legal_clause": len(b_q),
        "dimension_distribution": dict(dim_counter),
        "law_distribution": dict(law_counter),
        "difficulty_distribution": dict(diff_counter),
    }
    with open(OUT := HERE / "_questions_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps(s, ensure_ascii=False, indent=2))
