# Xiaoyi Enterprise Legal RAG — Retrieval Evaluation V1 (Offline A/B)

> 核心问题：**Chunk Strategy B（结构感知混合分块）是否比 Strategy A（滑窗基线）更适合企业法律合同 RAG 检索？**
> 本报告仅使用离线纯稠密检索；禁止 Milvus / Reranker / Hybrid Search / FastAPI 启动。

**Final Gate**: `RETRIEVAL_EVAL_V1 = PASS`

- REVIEW_REQUIRED: `False`
- Gate Reason: PASS conditions met.

## 1. 背景与冻结边界

- **上游冻结**：Parser V2.4.2 / Cleaner V2.1 / QGate Policy V2.4 / Chunk Metadata Contract V1 / Chunking Strategy V2 均未修改。
- **Preferred Chunk Strategy B**: Structure-aware Hybrid Chunking (target=600 chars / hard=900)
- **Embedding 模型**：项目已配置本地 `models/bge-m3`（dim=1024, normalize_embeddings=True, CPU）
- **A 类查询**：50 条合同事实查询（基于 10 份 SYN_CONTRACT_*.docx，synthetic_contract_facts.jsonl 仅作为 GT，未进入语料/向量）
- **B 类查询**：50 条法律条款查询（20 条来自 legal_eval_v1.jsonl anchor + 30 条从民法典/个保法/公司法 canonical chunks 程序化生成）
- **检索算法**：纯稠密 cosine similarity（点积），top_k ∈ {5,10}，tie-break 按 chunk_id 升序

## 2. 数据与分块规模

| Item | Strategy A (Sliding Window) | Strategy B (Structure-aware) |
|---|---:|---:|
| Chunk 总数 (CANON + SYN) | 2250 | 4599 |
|   - CANON canonical-53 | 1849 | 4145 |
|   - SYN synthetic-10 合同 | 401 | 454 |
| Embedding 维度 | 1024 | 1024 |
| Build 耗时 (CPU) | - | 5626.3 s |

### Questions 分布摘要

- **total**: 100
- **contract_fact**: 50
- **legal_clause**: 50
- **dimension_distribution**:
  - contract_amount: 6
  - party_info: 5
  - sign_dates: 6
  - breach_terms: 5
  - tax_rate: 5
  - auto_renewal: 5
  - payment_terms: 4
  - governing_law: 5
  - termination_notice: 4
  - product_qty: 5
- **law_distribution**:
  - LAW_008: 9
  - LAW_009: 3
  - LAW_010: 2
  - LAW_007: 16
  - LAW_004: 1
  - LAW_005: 1
  - LAW_006: 1
  - LAW_001: 13
  - LAW_002: 2
  - LAW_003: 1
  - DATA_002: 1
- **difficulty_distribution**:
  - easy: 30
  - hard: 15
  - medium: 55
- **citation_evidence_fallback_rate**: 0.05
- **queries_with_ce_hit_top10**: 83
- **queries_with_ce_hit_all_fallback_top10**: 5

## 3. 核心指标汇总

| STRATEGY | QUERY_SCOPE | NUM_QUERIES | HIT_RATE@5 | RECALL@5 | HIT_RATE@10 | RECALL@10 | MRR@10 | CITATION_EV_HIT_RATE@10 |
|---|---|---|---|---|---|---|---|---|
| A | ALL | 100 | 0.660000 | 0.660000 | 0.770000 | 0.770000 | 0.415456 | 0.770000 |
| A | CONTRACT_FACT | 50 | 0.380000 | 0.380000 | 0.560000 | 0.560000 | 0.199913 | 0.560000 |
| A | LEGAL_CLAUSE | 50 | 0.940000 | 0.940000 | 0.980000 | 0.980000 | 0.631000 | 0.980000 |
| B | ALL | 100 | 0.680000 | 0.680000 | 0.780000 | 0.780000 | 0.536381 | 0.780000 |
| B | CONTRACT_FACT | 50 | 0.440000 | 0.440000 | 0.580000 | 0.580000 | 0.302762 | 0.580000 |
| B | LEGAL_CLAUSE | 50 | 0.920000 | 0.920000 | 0.980000 | 0.980000 | 0.770000 | 0.980000 |

### 3.1 Δ (B − A)：95% Bootstrap 置信区间 (R=1000, seed=20260821)

| Metric | Δ mean | 95% CI low | 95% CI high |
|---|---:|---:|---:|
| RECALL@10 | +0.0086 | -0.0600 | +0.0800 |
| MRR@10 | +0.1215 | +0.0496 | +0.1944 |
| CITATION_EV_HIT_RATE@10 | +0.0116 | -0.0500 | +0.0700 |

> 解读：Recall@10 Δ = +0.0086，MRR@10 Δ = +0.1215。
> Strategy B 的 Recall@10 **不低于** Strategy A，符合主闸门 PASS 条件。

### 3.2 分项差异（A 类合同事实 vs B 类法律条款）

#### Scope: CONTRACT_FACT

| 指标 | A | B | Δ |
|---|---:|---:|---:|
| RECALL@10 | 0.5600 | 0.5800 | +0.0200 |
| MRR@10 | 0.1999 | 0.3028 | +0.1028 |
| CITATION_EV_HIT_RATE@10 | 0.5600 | 0.5800 | +0.0200 |
| RECALL@5 | 0.3800 | 0.4400 | +0.0600 |

#### Scope: LEGAL_CLAUSE

| 指标 | A | B | Δ |
|---|---:|---:|---:|
| RECALL@10 | 0.9800 | 0.9800 | +0.0000 |
| MRR@10 | 0.6310 | 0.7700 | +0.1390 |
| CITATION_EV_HIT_RATE@10 | 0.9800 | 0.9800 | +0.0000 |
| RECALL@5 | 0.9400 | 0.9200 | -0.0200 |

## 4. 典型失败/差异样例 (≥ 10 cases)

### Case 1. [A_HIT_B_MISS] CQ-004 (contract_fact, difficulty=hard)

- **Query**: 乙方逾期交付数据委托处理合同标的物的违约金率是多少？上限？
- **GT doc_ids**: `['CONTRACT_008', 'SYN-001', 'SYN_CONTRACT_001_DATA_PROC']`
- **Rank A**: 2, **Rank B**: None
- **A Top-1 snippet**: 违约金；累计违约金上限为合同金额的12%；甲方逾期付款按日万分之二支付滞纳金。
争议解决与管辖：因本合同发生之争议，由上海市浦东新区人民法院诉讼管辖。
关键条款事实：
- tax_rate：0.06
- subcontract：未经甲方书面…
- **B Top-1 snippet**: 违约条款：乙方逾期供货按日支付当批次未供货金额0.03%的违约金，上限15%；
产品不合格造成甲方工程损失的，乙方据实赔付。

### Case 2. [A_HIT_B_MISS] CQ-009 (contract_fact, difficulty=hard)

- **Query**: 数据委托处理合同的争议解决机构是？管辖法院？
- **GT doc_ids**: `['CONTRACT_008', 'SYN-002', 'SYN_CONTRACT_002_DATA_PROC']`
- **Rank A**: 7, **Rank B**: None
- **A Top-1 snippet**: 反合同约定造成原始数据的重大泄露或滥用事件的。
5.有下列情形之一的，乙方有权单方解除本合同：
（1）甲方要求委托处理的原始数据在合法性、合规性、权属等方面存在重大问题的。
（2）甲方未履行合同主要义务且经乙方催告仍不履行义务超过 个工作日…
- **B Top-1 snippet**: （六）数据使用权：是指权利人通过加工、聚合、分析等方式，将数据用于优化生产经营、提供社会服务、形成衍生数据等的权利。一般来说，使用权是权利人在不对外提供数据的前提下，将数据用于内部使用的权利。
（七）数据经营权：是指权利人通过转让、许可、出…

### Case 3. [A_HIT_B_MISS] CQ-014 (contract_fact, difficulty=hard)

- **Query**: 本数据中介服务合同约定的纠纷解决方式是诉讼还是仲裁？具体机构？
- **GT doc_ids**: `['CONTRACT_006', 'SYN-003', 'SYN_CONTRACT_003_DATA_AGENCY']`
- **Rank A**: 8, **Rank B**: None
- **A Top-1 snippet**: ，任何一方不得将本合同项下的权利或义务转让给第三方。
3.因不可抗力情形致使合同无法继续履行的，本合同自不可抗力发生之日起解除。当事人主张解除合同的，应当及时通知其他当事人。
4.有下列情形之一的，乙方有权单方解除合同：
（1）标的数据的合…
- **B Top-1 snippet**: 人民法院或者仲裁机构应当结合案件的实际情况 ， 根据公平原则变更或
者解除合同 。

### Case 4. [A_HIT_B_MISS] CQ-002 (contract_fact, difficulty=easy)

- **Query**: XH-YS-SYN-2026-0418-001合同的甲方（委托方）是谁？
- **GT doc_ids**: `['CONTRACT_008', 'SYN-001', 'SYN_CONTRACT_001_DATA_PROC']`
- **Rank A**: 4, **Rank B**: None
- **A Top-1 snippet**: 0512-SYN-0003 [卖方/乙方]广东省深圳市南山区科技园南区 SYN-19 栋海岳大厦 8 层（虚构） 0755-SYN-0004 |
| 乙方收款信息 | 开户行 | |
| 乙方收款信息 | 账号 | |
第五条 双方义务
5…
- **B Top-1 snippet**: 第十八条 其他
。
甲方（买受人，签名/盖章）：杭州云杉数据服务有限公司（SYNTHETIC） 乙方（出卖人，签名/盖章）：成都远川工程管理有限公司（SYNTHETIC）
甲方法定代表人：孟惊鸿（SYNTHETIC） 乙方法定代表人：程雪楼…

### Case 5. [A_HIT_B_MISS] CQ-041 (contract_fact, difficulty=easy)

- **Query**: 2026-05-20签订的本合同，乙方为哪家公司？
- **GT doc_ids**: `['CONTRACT_017', 'SYN-009', 'SYN_CONTRACT_009_EPC']`
- **Rank A**: 8, **Rank B**: None
- **A Top-1 snippet**: nd Truth 事实附录（SYNTHETIC）】
synthetic_contract_id：SYN-006
合同类型：PROCUREMENT_MATERIAL
合同编号：QM-HY-SYN-2026-0515-006
甲方：苏州启明智能…
- **B Top-1 snippet**: 第十八条 其他
。
甲方（买受人，签名/盖章）：杭州云杉数据服务有限公司（SYNTHETIC） 乙方（出卖人，签名/盖章）：成都远川工程管理有限公司（SYNTHETIC）
甲方法定代表人：孟惊鸿（SYNTHETIC） 乙方法定代表人：程雪楼…

### Case 6. [B_HIT_A_MISS] CQ-015 (contract_fact, difficulty=hard)

- **Query**: 数据中介服务合同重要条款中关于标的物数量与比例的说明？
- **GT doc_ids**: `['CONTRACT_006', 'SYN-003', 'SYN_CONTRACT_003_DATA_AGENCY']`
- **Rank A**: None, **Rank B**: 7
- **A Top-1 snippet**: 大写数字为准。
六、本合同文本未尽事项，可由当事人附页另行约定，并作为本合同的组成部分。
七、名词解释：
（一）标的数据：是指本合同约定的中介服务对应的一系列数据。
（二）数据中介服务：是指依据合同约定，中介方向委托方提供的市场推广、客户对…
- **B Top-1 snippet**: 六、本合同文本未尽事项，可由当事人附页另行约定，并作为本合同的组成部分。
七、名词解释：
（一）标的数据：是指本合同约定的中介服务对应的一系列数据。
（二）数据中介服务：是指依据合同约定，中介方向委托方提供的市场推广、客户对接、合同订立等交…

### Case 7. [B_HIT_A_MISS] CQ-024 (contract_fact, difficulty=hard)

- **Query**: 请查询材料采购合同的管辖法院/仲裁条款。
- **GT doc_ids**: `['CONTRACT_111', 'SYN-005', 'SYN_CONTRACT_005_PROC_MAT']`
- **Rank A**: None, **Rank B**: 3
- **A Top-1 snippet**: 付材料款（大写）元。
第十二条 报酬及材料费的结算方式及期限： 第十三条 保修期限： 第十四条 本合同解除的条件： 第十五条 违约责任：
第十六条合同争议的解决方式：本合同在履行过程中发生的争议，由双方当事人协商解 决；也可由当地工商行政管…
- **B Top-1 snippet**: 第三十五条　合同或者其他财产权益纠纷的当事人可以书面协议选择被告
住所地、合同履行地、合同签订地、原告住所地、标的物所在地等与争议有实际
联系的地点的人民法院管辖，但不得违反本法对级别管辖和专属管辖的规定。

### Case 8. [B_HIT_A_MISS] CQ-050 (contract_fact, difficulty=hard)

- **Query**: 本合同重要事实中的量化指标有哪些（百分比/吨/保底金额等）？
- **GT doc_ids**: `['CONTRACT_005', 'SYN-010', 'SYN_CONTRACT_010_ENTRUST']`
- **Rank A**: None, **Rank B**: 9
- **A Top-1 snippet**: .00 | 637,160.00 | 82,840.00 | 720,000.00 |
| 2 | 矿渣粉 S95 | GB/T 18046-2017 | 海岳供应链下属合作工厂（SYN） | GB/T 18046-2017 | 1000 …
- **B Top-1 snippet**: 合同金额（含税，元）：人民币 1,250,000 元（大写：壹佰贰拾伍万元整（SYNTHETIC））
币种：人民币（CNY）
付款条件与方式：结果数据验收合格后30个工作日内支付80%即1,000,000元，质量保证期届满30个工作日内支付…

### Case 9. [B_HIT_A_MISS] CQ-031 (contract_fact, difficulty=easy)

- **Query**: 合同编号YS-YC-SYN-2026-0320-007中，乙方名称是？
- **GT doc_ids**: `['CONTRACT_003', 'SYN-007', 'SYN_CONTRACT_007_SALE_AGRI']`
- **Rank A**: None, **Rank B**: 1
- **A Top-1 snippet**: 0512-SYN-0003 [卖方/乙方]广东省深圳市南山区科技园南区 SYN-19 栋海岳大厦 8 层（虚构） 0755-SYN-0004 |
| 乙方收款信息 | 开户行 | |
| 乙方收款信息 | 账号 | |
第五条 双方义务
5…
- **B Top-1 snippet**: 第十八条 其他
。
甲方（买受人，签名/盖章）：杭州云杉数据服务有限公司（SYNTHETIC） 乙方（出卖人，签名/盖章）：成都远川工程管理有限公司（SYNTHETIC）
甲方法定代表人：孟惊鸿（SYNTHETIC） 乙方法定代表人：程雪楼…

### Case 10. [B_HIT_A_MISS] CQ-034 (contract_fact, difficulty=medium)

- **Query**: YS-YC-SYN-2026-0320-007合同提前解除的通知期限是多少日？
- **GT doc_ids**: `['CONTRACT_003', 'SYN-007', 'SYN_CONTRACT_007_SALE_AGRI']`
- **Rank A**: None, **Rank B**: 2
- **A Top-1 snippet**: 日期：2026-05-01
到期日期：2028-06-30
自动续约：是（是，续约期1年）
续约期限：1年
提前解除通知期限：90日
合同金额（含税，元）：人民币 1,250,000 元（大写：壹佰贰拾伍万元整（SYNTHETIC））
币种…
- **B Top-1 snippet**: 提前解除通知期限：30日
合同金额（含税，元）：人民币 450,000 元（大写：肆拾伍万元整（SYNTHETIC））
币种：人民币（CNY）
付款条件与方式：乙方促成甲方与数据需求方签约并完成验收后30日内，甲方按实际成交金额的3.2%向…

### Case 11. [A_BETTER_RANK] CQ-017 (contract_fact, difficulty=medium)

- **Query**: 请问数据中介服务合同期满是否会自动续签？若续约，续多久？
- **GT doc_ids**: `['CONTRACT_006', 'SYN-004', 'SYN_CONTRACT_004_DATA_AGENCY']`
- **Rank A**: 4, **Rank B**: 9
- **A Top-1 snippet**: 条 其他规定
1.本合同自双方签字并盖章之日起生效，有效期至 年 月 日止。
2.本合同一式 份，各方各执 份，具有同等法律效力。
3.本合同中部分条款被认定为无效或不可执行，不影响合同其他条款。
4.本合同构成双方关于数据中介服务的完整协…
- **B Top-1 snippet**: 第十一条 其他规定
1.本合同自双方签字并盖章之日起生效，有效期至 年 月 日止。
2.本合同一式 份，各方各执 份，具有同等法律效力。
3.本合同中部分条款被认定为无效或不可执行，不影响合同其他条款。
4.本合同构成双方关于数据中介服务的…

### Case 12. [A_BETTER_RANK] CQ-019 (contract_fact, difficulty=medium)

- **Query**: 数据中介服务合同终止条款中，解约提前通知期多长？
- **GT doc_ids**: `['CONTRACT_006', 'SYN-004', 'SYN_CONTRACT_004_DATA_AGENCY']`
- **Rank A**: 2, **Rank B**: 7
- **A Top-1 snippet**: ，任何一方不得将本合同项下的权利或义务转让给第三方。
3.因不可抗力情形致使合同无法继续履行的，本合同自不可抗力发生之日起解除。当事人主张解除合同的，应当及时通知其他当事人。
4.有下列情形之一的，乙方有权单方解除合同：
（1）标的数据的合…
- **B Top-1 snippet**: 提前解除通知期限：30日
合同金额（含税，元）：人民币 450,000 元（大写：肆拾伍万元整（SYNTHETIC））
币种：人民币（CNY）
付款条件与方式：乙方促成甲方与数据需求方签约并完成验收后30日内，甲方按实际成交金额的3.2%向…

### Case 13. [A_BETTER_RANK] CQ-044 (contract_fact, difficulty=medium)

- **Query**: 本合同终止条款中，解约提前通知期多长？
- **GT doc_ids**: `['CONTRACT_017', 'SYN-009', 'SYN_CONTRACT_009_EPC']`
- **Rank A**: 4, **Rank B**: 9
- **A Top-1 snippet**: 日期：2026-05-01
到期日期：2028-06-30
自动续约：是（是，续约期1年）
续约期限：1年
提前解除通知期限：90日
合同金额（含税，元）：人民币 1,250,000 元（大写：壹佰贰拾伍万元整（SYNTHETIC））
币种…
- **B Top-1 snippet**: 提前解除通知期限：30日
合同金额（含税，元）：人民币 450,000 元（大写：肆拾伍万元整（SYNTHETIC））
币种：人民币（CNY）
付款条件与方式：乙方促成甲方与数据需求方签约并完成验收后30日内，甲方按实际成交金额的3.2%向…

### Case 14. [A_BETTER_RANK] CQ-048 (contract_fact, difficulty=medium)

- **Query**: 是否自动续约条款查询。
- **GT doc_ids**: `['CONTRACT_005', 'SYN-010', 'SYN_CONTRACT_010_ENTRUST']`
- **Rank A**: 2, **Rank B**: 5
- **A Top-1 snippet**: 日期：2026-05-01
到期日期：2028-06-30
自动续约：是（是，续约期1年）
续约期限：1年
提前解除通知期限：90日
合同金额（含税，元）：人民币 1,250,000 元（大写：壹佰贰拾伍万元整（SYNTHETIC））
币种…
- **B Top-1 snippet**: 自动续约：否，合同期满自动终止，不再续约；
如需续约双方另行签订书面合同。
提前解除通知期限：60日；
违约解除违约金：2个月租金。
违约责任：乙方逾期支付租金按日万分之三付滞纳金；
甲方逾期交房按日万分之三付违约金；
乙方擅自转租的，甲方…

### Case 15. [A_BETTER_RANK] LQ-023 (legal_clause, difficulty=medium)

- **Query**: 个人信息保护法第三条的主要内容是什么？
- **GT doc_ids**: `['LAW_001']`
- **Rank A**: 1, **Rank B**: 4
- **A Top-1 snippet**: int())【纠错】
中华人民共和国个人信息保护法
（2021年8月20日第十三届全国人民代表大会常务委员会第三十次会议通过）
目　　录
第一章　总　　则
第二章　个人信息处理规则
第一节　一般规定
第二节　敏感个人信息的处理规则
第三节　…
- **B Top-1 snippet**: 中华人民共和国个人信息保护法
2021年08月20日 21:21
来源：
中国人大网
[](#)[](#)
 Baidu Button BEGIN
[](#) [](#)
 Baidu Button END [【打印】](javascrip…

### Case 16. [B_BETTER_RANK] CQ-030 (contract_fact, difficulty=hard)

- **Query**: 材料采购合同项下的产品/服务数量是多少？（吨/千克/件）
- **GT doc_ids**: `['CONTRACT_111', 'SYN-006', 'SYN_CONTRACT_006_PROC_MAT']`
- **Rank A**: 9, **Rank B**: 2
- **A Top-1 snippet**: 通。
（六）变更。
十一、其他要求
（一）对承包人的主要人员资格要求。
（二）相关审批、核准和备案手续的办理。
（三）对项目业主人员的操作培训。
（四）分包。
（五）设备供应商。
（六）缺陷责任期的服务要求。
附件 2
发包人供应材料设备一…
- **B Top-1 snippet**: 一、合同的组成
以下文件是本合同不可分割的组成部分，如果不同文件的条款之间有冲突，文件之间的优先效力顺序如下：
1.本合同及其附件、补充协议；
2.中标通知书（含乙方承诺函等）；
3.乙方提供的投标文件（含澄清文件及承诺等）；
4.甲方发出…

### Case 17. [B_BETTER_RANK] CQ-006 (contract_fact, difficulty=easy)

- **Query**: 该合同总价为多少人民币？合同款额是多少？
- **GT doc_ids**: `['CONTRACT_008', 'SYN-002', 'SYN_CONTRACT_002_DATA_PROC']`
- **Rank A**: 10, **Rank B**: 2
- **A Top-1 snippet**: 价（含税）：人民币 12,000,000 元（大写：壹仟贰佰元整（SYNTHETIC），税率 9%）
付款安排：签约合同价含税12,000,000元；签约后10日内付10%预付款=1,200,000元；按月计量进度款（当月审核后30日内付至…
- **B Top-1 snippet**: 合同金额（含税，元）：人民币 3,920,000 元（大写：叁佰玖拾贰万元整（SYNTHETIC））
币种：人民币（CNY）
付款条件与方式：甲方按季度预付固定佣金人民币980,000元整/季；
每季度开始前5个工作日支付；
首年合计3,9…

### Case 18. [B_BETTER_RANK] CQ-008 (contract_fact, difficulty=medium)

- **Query**: 甲方向乙方支付数据委托处理合同款项的付款条件和时间节点？
- **GT doc_ids**: `['CONTRACT_008', 'SYN-002', 'SYN_CONTRACT_002_DATA_PROC']`
- **Rank A**: 6, **Rank B**: 3
- **A Top-1 snippet**: 甲方一次性向乙方支付全部费用。
3.乙方应依法向甲方开具等额有效的增值税□专用□普通发票。如因国家税务政策变更等导致税率变化，应以变化后的税率为准。
4.乙方收款账户信息：
账户名称： ；
开户银行： ；
账 号： 。
5.甲方开票信息：
…
- **B Top-1 snippet**: 第五条 付款方式
（一）甲方按照以下第 种方式支付委托费用。
1.一次性付款，付款时间为： 。
2.分期付款，付款时间及数额分别为：
（1） ；
（2） ；
（3） 。
（二）乙方收款账户信息：
户名： ；
账号： ；
开户行： 。

### Case 19. [B_BETTER_RANK] CQ-013 (contract_fact, difficulty=medium)

- **Query**: 任何一方要求不再续约或提前解除需提前多少天通知？
- **GT doc_ids**: `['CONTRACT_006', 'SYN-003', 'SYN_CONTRACT_003_DATA_AGENCY']`
- **Rank A**: 10, **Rank B**: 1
- **A Top-1 snippet**: 表示或者以自己的行为表明
不履行主要债务 ；
（ 三 ） 当事人一方迟延履行主要债务 ， 经催告后在合理期限内仍未履行 ；
（ 四 ） 当事人一方迟延履行债务或者有其他违约行为致使不能实现合同
目的 ；
（ 五 ） 法律规定的其他情形 。
…
- **B Top-1 snippet**: 提前解除通知期限：30日
合同金额（含税，元）：人民币 450,000 元（大写：肆拾伍万元整（SYNTHETIC））
币种：人民币（CNY）
付款条件与方式：乙方促成甲方与数据需求方签约并完成验收后30日内，甲方按实际成交金额的3.2%向…

### Case 20. [B_BETTER_RANK] CQ-032 (contract_fact, difficulty=medium)

- **Query**: YS-YC-SYN-2026-0320-007合同约定的续约条款是怎样的？是否自动续期？
- **GT doc_ids**: `['CONTRACT_003', 'SYN-007', 'SYN_CONTRACT_007_SALE_AGRI']`
- **Rank A**: 8, **Rank B**: 1
- **A Top-1 snippet**: 日期：2026-05-01
到期日期：2028-06-30
自动续约：是（是，续约期1年）
续约期限：1年
提前解除通知期限：90日
合同金额（含税，元）：人民币 1,250,000 元（大写：壹佰贰拾伍万元整（SYNTHETIC））
币种…
- **B Top-1 snippet**: 自动续约：否
提前解除通知期限：15日
违约责任：乙方交货质量不合格按当批次货款的20%支付违约金，并应在3日内免费调换；
甲方逾期付款按日万分之三支付滞纳金。
争议解决：因本合同发生之争议，由杭州市余杭区人民法院诉讼管辖。
仓储要求：无冷…

## 5. 冻结边界核对 (Checklist)

- ✅ modules.milvus_store: imported? = False
- ✅ modules.database: imported? = False
- ✅ modules.cache: imported? = False
- ✅ modules.rerank: imported? = False
- ✅ modules.rag.hybrid_rrf: imported? = False
- ✅ pymilvus: imported? = False

## 6. 结论与下一步

**主闸门**：`RETRIEVAL_EVAL_V1 = PASS`

结论：Strategy B 在全量 scope 的 Recall@10 不劣于 Strategy A，且 Citation Evidence 损失在阈值以内（或更优）。
推荐后续：将 Strategy B 推送到在线链路候选；并开展真实用户 query 的在线 A/B（配合 reranker/Hybrid）。

**本次评估严格停止在 Retrieval Evaluation V1 完成处；未进行任何 Embedding 写 Milvus / MySQL / Redis / Rerank / FastAPI 启动。**