# 翻译模块问题与重写指南

> 状态：Step 1–8D 已验收；Step 9 真实 Provider 路径已实现，等待凭据后执行付费端到端验收  
> 范围：PDF 解析结果到结构化译文的生成，以及翻译与 Reflow 之间的数据边界  
> 当前不包含：完整生产级排版引擎重写；真实 LLM 付费验收尚未执行

## 1. 目标

将现有的翻译原型重写为一个独立、可测试、可恢复、可观测的 Translation Stage，并与解析和 Reflow 通过明确的 JSON 契约连接。

重写后应当满足：

1. 只翻译明确允许翻译的元素，不改写 URL、公式、页码等结构内容。
2. 同一个逻辑段落只翻译一次，跨 bbox 文本不被割裂成无上下文请求。
3. 公式、引用、编号和专有符号可验证地保持不变。
4. 译文与排版结果解耦，布局调整不得静默改变译文语义。
5. API 调用可缓存、可重试、可断点续跑，并可追踪模型、耗时、token 与错误。
6. 任何新类型都必须先进入类型策略表，不能默认送给翻译模型。

## 2. 当前实际执行逻辑

当前较完整的路径是 `external/Colayout-pure/run_translation.py`：

```text
解析 JSON + origin.pdf
  -> 补充 rich_spans
  -> 按 logical_para_id 聚合
  -> 基于 PDF 字号和坐标修正元素类型
  -> 公式占位符保护
  -> 逐逻辑段落调用远程翻译 API
  -> 英文残留/提示词泄漏/公式安全检查
  -> 失败修复与重译
  -> 恢复公式
  -> 按原 bbox 容量将译文拆回多个块
  -> 写入 translated/golden_style/detected_type/translation_guard
  -> Reflow 测量、必要时再改写、渲染 PDF
```

当前配置中 `m0_plain_translation: true` 优先于 `enable_isolength: true`，所以实际初译使用 `TranslationAgentNoIsoLength`。

## 3. 建议的模块边界

### 3.1 Translation Stage 应负责

- 读取并校验解析 JSON。
- 根据统一的元素类型策略决定 translate/preserve/skip。
- 按 `logical_para_id` 建立翻译单元。
- 保护公式、引用、URL、标识符和不可变片段。
- 构建文档级术语表和有限上下文。
- 调用翻译 Provider，校验结果并记录质量状态。
- 保存“逻辑段落级译文”作为唯一语义真值。
- 使用与 Reflow 共享的 `LayoutMetrics` 和黄金样式参数，将逻辑段落译文切分到固定 bbox。
- 若在允许的字号/行距/字距范围内仍无法容纳，生成受约束重译并重新切分。
- 导出 Reflow 可直接消费的逐块 `translated` 和样式字段，但不负责最终 PDF 绘制。

### 3.2 Reflow Stage 应负责

- 读取译文和原 bbox。
- 使用与 Translation 共享的字体选择、字号、行距、字距和文本测量规则。
- 按 Translation 已生成的逐 bbox 译文片段进行绘制，不再负责常规切分。
- 检测溢出、留白、重叠和跨栏异常。
- bbox 大小不可调整；只允许在黄金参数边界内微调行距、字距和缩小字号。
- 在允许的排版动作后仍溢出时，向 Translation Stage 请求新的受约束候选译文及切分结果。
- 记录逻辑段落译文、实际显示片段和译文版本的映射。

### 3.3 共享但不应重复实现的能力

- 版本化 JSON schema。
- 元素类型 taxonomy。
- 稳定的 `block_id` 和 `logical_para_id`。
- 逻辑段落与多 bbox 的映射。
- 黄金样式参数和各层级允许的调整边界。
- 唯一的文字测量与溢出判定实现 `LayoutMetrics`。
- 公式/不可变片段的标识和校验结果。
- 译文状态和 Reflow 状态枚举。

## 4. 建议的翻译 JSON 契约

建议不覆盖解析产物，而是生成独立的 `*_translated.json`。为兼容现有 Reflow，第一阶段仍保留顶层 `translated` 字段，同时增加结构化元数据。

```json
{
  "schema_version": "translation-v3",
  "document": {
    "document_id": "...",
    "source_pdf": "...",
    "source_json": "...",
    "source_hash": "...",
    "source_language": "en",
    "target_language": "zh-CN"
  },
  "translation_run": {
    "provider": "...",
    "model": "...",
    "prompt_version": "...",
    "started_at": "...",
    "completed_at": "..."
  },
  "layout_status": "ready_for_reflow",
  "layout_run": {
    "contract_version": "translation-reflow-v2",
    "optimizer_version": "layout-optimizer-v2",
    "measurement_engine": "layout-metrics-v2"
  },
  "translation_units": [
    {
      "translation_unit_id": "tu-...",
      "logical_para_id": 12,
      "element_type": "text",
      "policy": "translate_text",
      "block_ids": ["p0001-b0003", "p0001-b0004"],
      "source_text": "source logical paragraph",
      "translation": {
        "status": "translated",
        "initial_text": "初译的完整逻辑段落",
        "candidates": [],
        "selected_text": "当前采用的完整逻辑段落",
        "selected_version": "initial",
        "source_hash": "...",
        "quality_flags": [],
        "protected_tokens": [],
        "execution": {
          "attempts": 1,
          "cache_hit": false
        }
      }
    }
  ],
  "items": [
    {
      "block_id": "p0001-b0003",
      "type": "text",
      "text": "original block text",
      "bbox": [10, 20, 100, 40],
      "translation_unit_id": "tu-...",
      "translation_status": "translated",
      "translated": "分配到该固定边界框的译文",
      "recommended_style": {
        "font_id": "zh_body_regular",
        "font_size_pt": 8.9,
        "line_height_multiplier": 1.18,
        "char_spacing_em": -0.01
      }
    }
  ]
}
```

其中：

- `initial_text` 始终保留第一次完整译文。
- `candidates` 按时间记录因容量溢出产生的受约束重译及原因。
- `selected_text` 是当前最终选中的逻辑段落译文；无重译时等于 `initial_text`。
- Step 5 的逻辑段落级完整译文只在 `translation_units` 保存一次，逐块使用 `translation_unit_id` 关联，不在每个 bbox 重复存储。
- Step 5 不提前写入逐 bbox `translated`；Step 6 完成样式分析后，`layout_status=pending_block_split` 明确表示还不可交给 Reflow。
- Step 7 完成共享测量和切分后，再向每个 item 写入 `translated`，为现有 Reflow 提供兼容。
- Step 7 已固定 `translation-reflow-v2`：`selected_text` 是语义真值，所有元素 `translated` 按阅读顺序严格拼接后必须与之完全一致。
- Step 7 已固定 `layout-optimizer-v2`：先按各固定内容框在黄金字号和行距下的有效排版容量确定切分目标，在容差窗口内再优先自然标点边界；公式和受保护内容不得被切开。样式与跨框分配搜索使用可证明不改变最优解的下界剪枝，不使用“首个可行即停止”。布局测量器在单次文档运行内缓存 PyMuPDF 字符单位宽度与分词结果，并保持与未缓存路径逐字段一致。
- 仅在缩短译文实际可能解决溢出时请求受约束重译；最小行高已高于 bbox，或不可变 URL/公式本身无法容纳时，直接标记 `needs_review`，避免无效 API 循环。

需要支持的状态至少包括：

- `translated`：翻译完成并通过校验。
- `preserved`：按策略保留原文，例如 URL、作者、页码。
- `skipped`：无需文本翻译，例如图片主体。
- `failed`：API 或校验失败，不允许伪装成成功。
- `needs_review`：有译文，但公式、术语或完整性校验未通过。

## 5. 元素类型策略草案

| 类型 | 默认策略 | 允许长度改写 | 备注 |
|---|---|---:|---|
| `text` / `abstract` | 翻译 | 可配置 | 保留引用、公式和编号 |
| `title` / `section_header` | 翻译 | 谨慎 | 保留节号，优先简洁 |
| `caption` / `image_caption` / `figure_caption` / `table_caption` | 翻译 | 谨慎 | 保留 Figure/Table 编号；优先使用 `image_caption` |
| `header` / `footer` / `page_footnote` | 翻译 | 否 | 需明确页脚注和页脚的差异 |
| `author` / `affiliation` | 保留 | 否 | 可增加“机构名翻译”可选策略 |
| `reference` | 保留 | 否 | 可只翻译文献标题，但需单独解析 |
| `url` | 保留 | 否 | 严禁发送给模型改写 |
| `page_number` | 保留 | 否 | 严禁改写 |
| `equation` | 保留 | 否 | 公式翻译不属于正文翻译 |
| `image` | 跳过 | 否 | caption 应作为独立块翻译 |
| `table` | 翻译 HTML 文本节点 | 否 | 保留表格标签、属性、数字和统计符号 |
| 未知类型 | `needs_review` | 否 | 不得默认翻译 |

## 6. 问题与修正清单

### P0：实施重写前必须解决

- [x] **确定唯一翻译入口。**
  - 问题：`legacy/scripts/translate_and_backfill.py` 的导入路径已失效，与 `external/Colayout-pure/run_translation.py` 行为不一致。
  - 修正：建立一个新的标准 CLI，其他入口只作为兼容包装或删除。
  - 验收：`--check`、`--dry-run`、单篇翻译和批量翻译均从同一入口执行。

- [x] **移除代码库中的明文 API 密钥。**
  - 问题：`config.yaml` 被 Git 跟踪且存在非占位符密钥。
  - 修正：使用环境变量或本地私密覆盖文件；仓库仅保留 example config。
  - 验收：仓库扫描不包含有效密钥；启动时仅显示脱敏配置。

- [x] **确定唯一 JSON 数据来源和发现规则。**
  - 问题：当前通配符会同时命中适配后 JSON 和 MinerU 原生备份。
  - 建议：命令行显式接受 JSON 文件；传入目录时，优先 `*_enriched.json`，其次是适配后 `*_content_list.json`，永不自动选择 `*_mineru_content_list.json`。
  - 验收：日志明确记录 `source_json`、`source_pdf` 和选择原因；多候选且无法唯一时拒绝执行。

- [x] **定义并校验版本化数据契约。**
  - 问题：现有 Agent 隐式依赖 `type/text/page_idx/bbox/logical_para_id`。
  - 修正：使用 Pydantic 或 JSON Schema 验证，缺少必要字段时在 API 调用前失败。
  - 验收：坐标越界、页号越界、重复 ID、缺少逻辑段落 ID 都有可操作错误。

- [x] **将“翻译成功”与“保留原文/翻译失败”分开。**
  - 问题：多个异常分支用原文回退，但下游可能将其当成成功译文。
  - 修正：强制使用 `translated/preserved/failed/needs_review` 状态。
  - 验收：任何原文回退都必须有明确原因，不计入翻译成功率。

### P1：影响翻译正确性

- [x] **建立统一元素类型策略表。**
  - 修正：用单一映射表代替 `TARGET_TYPES/IGNORE_TYPES/SKIP_TEXT_TYPES` 多套分散集合。
  - 验收：`url/page_footnote/equation/table/image/reference`等都有唯一可测试策略。

- [x] **让目标语言真正可配置。**
  - 问题：参数名为 `target_lang`，但提示词和直接映射硬编码为中文。
  - 修正：使用标准 BCP-47 语言标识，提示词模板和质量检查按语言选择。
  - 验收：至少对 `en -> zh-CN` 有明确限定；不支持的语言对拒绝执行。

- [x] **扩展不可变内容保护。**
  - 修正：使用 tokenizer 同时保护 `$...$`、`$$...$$`、`\(...\)`、`\[...\]`、LaTeX environment、`\cite`、`\ref`、URL、DOI和代码片段。
  - 验收：保护 token 的数量、顺序和内容必须完全一致；不一致时不发布译文。

- [ ] **引入文档级术语一致性。**
  - 修正：从标题、摘要和高频术语生成可审计 glossary，后续段落带精简术语约束。
  - 验收：同一术语在文档内不应出现未记录的多种译法。

- [x] **将译文质量检查与排版检查分开。**
  - 翻译检查：完整性、语言、不可变 token、明显提示词泄漏、空响应。
  - 排版检查：溢出、留白、行数、字号、重叠。
  - 验收：排版失败不会把语义正确的 canonical translation 标记为翻译失败。

- [ ] **定义受约束重译协议。**
  - 修正：Reflow 不直接自由改写；它提交 `source_text/current_translation/max_width/max_lines/required_reduction_ratio/protected_tokens`，翻译模块返回新的有版本候选译文。
  - 验收：所有改写都可追溯到触发原因和前后文本。

### P2：影响稳定性、成本和可维护性

- [ ] **重写 API Client。**
  - 支持 429 `Retry-After`、5xx 重试、永久错误分类、连接/读取超时、请求 ID 和脱敏日志。
  - 温度、超时、并发度和最大输出 token 必须从有效配置读取。

- [x] **增加内容哈希缓存。**
  - cache key 至少包含：原文、源/目标语言、glossary 版本、prompt 版本、provider 和 model。
  - 不应将 API 密钥放入哈希或输出。

- [x] **增加断点恢复和原子写入。**
  - 每个逻辑段落完成后更新 manifest/checkpoint，写入临时文件后原子替换。
  - 重启时仅重译未完成或 source hash 已变更的单元。

- [ ] **使用受控并发，而非整篇粗粒度并发。**
  - 每个 provider 设置信号量、QPS 和 token 预算。
  - 结果按稳定 ID 写回，不依赖完成顺序。

- [ ] **消除两个 TranslationAgent 的大量重复。**
  - 提取共用的 segmenter、protector、quality guard 和 provider。
  - “自然翻译”和“受约束翻译”应是策略，而不是两份复制 Agent。

- [ ] **增加运行级可观测性。**
  - 记录段落数、缓存命中、请求数、重试、失败、needs_review、token、费用和耗时。
  - 不得在日志中输出密钥或完整敏感原文。

## 7. 测试与验收清单

### 单元测试

- [x] 类型策略覆盖所有已知类型，未知类型拒绝默认翻译。
- [x] `$...$`、`$$...$$`、`\(...\)`、`\[...\]`、`\cite`、`\ref`、URL 和 DOI 可完全往返。
- [x] 占位符丢失、重复、乱序和模型改写被检出。
- [ ] 模型空响应、超时、429、5xx 和非法 JSON 均有确定状态。
- [x] `target_language` 不受支持时在请求前失败。
- [x] cache key 在 prompt/model/glossary 改变时必须改变。

### 契约测试

- [ ] MinerUAdapter 产物可直接通过 Translation Input Schema。
- [x] 翻译产物可被现有 Reflow 兼容层读取。
- [x] 原始 `text/bbox/page_idx/type/logical_para_id` 不被翻译模块改写。
- [x] 逻辑段落译文能完整映射回所有展示块。

### 端到端测试

- [ ] 电子文本 PDF、扫描 PDF、双栏论文、公式密集论文、图表密集论文各一份 fixture。
- [x] 翻译阶段中断后重启，已完成段落不再调用 Provider。
- [x] 重复运行同一输入，缓存命中时输出一致。
- [x] 翻译失败的段落在最终报告中可见，不被原文回退掩盖。
- [x] Reflow 调整后 `selected_text` 保持不变，除非产生并选中了有记录的受约束候选版本。

## 8. 建议实施顺序

1. **决策阶段**：确认 JSON 来源、类型策略、语言范围、表格/参考文献策略和 Reflow 是否允许改写。
2. **安全与入口**：处理密钥，建立唯一 CLI 与 preflight，不调用真实 API。
3. **数据契约**：实现 schema、loader、taxonomy 和 legacy adapter。
4. **纯翻译内核**：实现 segmentation、protection、provider、QA、cache 和 checkpoint。
5. **兼容现有 Reflow**：保留顶层 `translated`，使新翻译输出先能被现有 Reflow 使用。
6. **Reflow 边界改造**：移出语义改写，增加受约束重译协议和 `selected_text/translated` 映射。
7. **端到端验收**：固定 fixture、质量指标和回归产物。

## 9. 已确认的设计决策

### D1. JSON 数据来源

1. 用户显式传入 JSON 时，只使用该文件。
2. 传入解析目录时，优先 `*_enriched.json`。
3. 若不存在 enriched，使用适配后 `*_content_list.json`，必要时从 PDF 补充样式。
4. 从不自动使用 `*_mineru_content_list.json` 和 `*_content_list_v2.json`。

### D2. 初译和受约束重译

- 保留“所有段落初译即受容量约束”模式。
- 初译使用原段落全部固定 bbox 的可容纳容量。
- 初译在实际切分和测量后仍溢出时，生成更严格的受约束候选译文。
- `initial_text` 保留初译，`candidates` 记录重译历史，`selected_text` 保存当前选中的最终译文。

### D3. Translation 与 Reflow 的语义权限

- 多 bbox 译文切分由 Translation Stage 负责。
- Reflow 按 Translation 生成的逐 bbox `translated` 进行统一渲染和复核。
- Reflow 可在检测到溢出后请求 Translation Stage 生成受约束重译，但不直接静默改写译文。
- 公式在 Translation 写入结果 JSON 前恢复为原式，同时保留保护 token 元数据供校验。

### D4. 边界框和黄金样式

- 原 bbox 的坐标和大小绝对不可调整。
- 前期按全文不同文本层级统计字号、行距和字距，生成黄金参数。
- 允许在边界内微调行距和字距，以及缩小字号；不得过分偏离对应层级的黄金参数。
- 第一版字号下限：正文/摘要最多缩小 10%，标题/章节标题最多缩小 8%，caption/header/footer/page_footnote 最多缩小 12%。
- 第一版行距范围：对应层级黄金值的 -8%～+5%。
- 第一版字距范围：-0.04em～+0.02em。
- 黄金参数按“每份文档 × 文本层级”统计，优先使用稳健中位数；样本不足时的层级回退规则在 Step 6 固化并测试。
- Step 6 已固定 `layout-style-v1`：只使用原始 PDF/解析 `rich_spans`，不使用译文反推样式；字号按文字量加权的稳健中位数统计，异常值被过滤，缺失层级由正文按固定比例派生并记录 provenance。
- Step 6 已固定 `layout-metrics-v1`：Translation 和 Reflow 必须使用同一字体文件、字号、行距和字距进行测量。
- 在允许的排版参数内仍溢出时，才发起受约束重译。
- Translation 和 Reflow 必须共享同一个 `LayoutMetrics` 实现，避免测量与实际渲染不一致。

### D5. 翻译范围

- 第一版只支持 `English -> Simplified Chinese`。
- 参考文献全部保留原文。
- 作者姓名、邮箱、ORCID 和 URL 保留；机构名、城市和国家翻译。无法可靠拆分的作者块保留原文并标记 `needs_review`。
- 表格 caption 正常翻译。表格表头和单元格也需要翻译，但只改写 `table_body` HTML 文本节点，保留标签、属性、数字和统计符号，输出 `translated_table_body`。
- 表格主体和表格 caption 即使共享 `logical_para_id` 也必须建立两个独立翻译单元。
- 图片内部文字暂不属于本阶段范围；图片 caption 正常翻译，统一命名为 `image_caption`。
- 通用 `caption` 与同一 `logical_para_id` 的唯一 table 关联时归类为 `table_caption`，与唯一 image 关联时归类为 `image_caption`；两者都有或都没有时保留为通用 `caption` 并标记 `needs_review`。

### D6. Provider 和数据隐私

- 继续使用 DeepSeek/OpenAI 兼容 API。
- 支持每次运行显式选择一个 Provider；第一版不在同一文档内自动故障转移到另一个 Provider。
- 允许将论文正文发送给第三方 API。
- 本地翻译模型暂不属于第一版范围。

### D7. 输出形式

- 不原地修改解析 JSON。
- 独立输出 `*_translated.json` 和 `*_translation_manifest.json`。
- 第一版保留现有 Reflow 需要的逐块 `translated/golden_style/detected_type`。

## 10. 翻译与 Reflow 的分阶段改造

**结论：不需要同时全量重写，翻译可以先单独重写，但必须共享数据契约和排版测量实现。**

可行路径：

1. 先定义 Translation Output Schema。
2. 新翻译模块输出 `initial_text/candidates/selected_text` 和兼容字段。
3. Translation Stage 使用共享 `LayoutMetrics` 将 `selected_text` 切分到原有固定 bbox，写入逐块 `translated`。
4. 现有 Reflow 暂时继续读取逐块 `translated`，复核测量并完成渲染。
5. 仍溢出时，Reflow 通过明确协议向 Translation Stage 请求受约束重译和新的逐块切分。

Step 8 已将这一边界落地为“薄 Reflow”：

- 只消费 `translation-reflow-v1` 产物，不修改 `selected_text` 或逐块 `translated`。
- 使用共享 `layout-metrics-v1` 复核字体、字号、行距、字距与固定 bbox；测量不一致时拒绝渲染。
- 对已通过预检的文本仅清除原 bbox 内文字，保留图像、线条和页面几何后回填。
- 对仍溢出且确实可通过压缩译文解决的元素，只输出结构化受约束重译请求，不在 Reflow 内调用 LLM。
- 默认存在阻断项时不生成 PDF；显式使用 `--allow-partial` 时，只渲染已批准元素，其余保留原文并写入反馈报告。
- 输出 `reflow-feedback-v1` 报告，记录每个元素的状态、原因、测量、动作和可选重译请求。

### 10.1 公式矢量回填扩展（Step 8A–8D）

公式能力在既有 Translation/Reflow 边界上分步实现：

1. **Step 8A（已验收）：公式清单、来源和数据契约。**
   - 输出升级为 `translation-v2`，新增 `formula-contract-v1`；读取端仍兼容 `translation-v1`。
   - 每个公式都有稳定 `formula_id`、LaTeX 及哈希、所属翻译单元、原页面/块/行/span 路径与独立 `source_bbox`。
   - 使用顺序保持的 LaTeX 绑定，不凭空猜测 MinerU span 与文本占位符的对应关系。
   - 行内公式使用自身 span bbox；独立公式当前标记为原位保留，其英文说明暂不翻译。
   - 只有文本 LaTeX 而没有 MinerU span 时，明确选择 MathJax SVG 候选；只有 MinerU span 却没有受保护文本时，进入 `needs_review`。
   - MinerU 四舍五入造成的小幅 bbox 差异写入 `geometry_flags`；超过可接受容差的公式 bbox 仍在翻译前拒绝。
2. **Step 8B（已验收）：公式矢量资产引擎。**
   - 输出升级为 `translation-v3`，新增 `formula-assets-v1`；读取端仍兼容 `translation-v1/v2`。
   - 有可靠 MinerU span 时，检查原 PDF clip 中的文字、路径和位图内容，并保存绑定原 PDF 哈希的矢量 clip 定位器，不重复复制整页资源。
   - 行内 clip 的 PDF 文字必须与 LaTeX 可视签名一致；小幅夹带相邻行字符时可稳定缩紧 bbox，无法可靠分离时降级到 MathJax。
   - 无源 span 或源 clip 不可用时，使用项目本地固定版本 MathJax 4.1.3 批量生成自包含 SVG path，不依赖 CDN 或系统数学字体。
   - SVG 写入前校验 XML、拒绝 `script` 和外部引用，并记录哈希、viewBox、宽高、基线深度、生成引擎和降级原因。
   - 禁用 MathJax 4 默认的 inline 自动断行，保证一个公式对应一个 SVG 资产。
3. **Step 8C（已验收）：`LayoutMetrics v2` 混合 run 测量。**
   - `layout-metrics-v2` 将中文文字和行内公式建模为不可拆分的有序 run；公式参与真实宽度、行高、溢出和换行计算，不再按 LaTeX 源码字符宽度估算。
   - 原 PDF clip 按目标字号与原文层级字号的比例缩放；MathJax SVG 使用固定 `0.5em/ex` 换算，测量结果记录实际 point 尺寸和资产缩放系数。
   - 每行记录共享基线、实际行高和 run 的横向偏移；高公式可以抬高行高，文字和公式使用同一基线模型。
   - Translation 在多 bbox 切分时保证公式 run 不跨切分点，并将公式归属更新为译文实际落入的 bbox，而不是原文 bbox。
   - 输出升级为 `translation-reflow-v2`；`translation-reflow-v1` / `layout-metrics-v1` 旧产物仍可读取。
   - Reflow 已能独立复算并核对混合测量，但公式矢量绘制仍明确阻断到 Step 8D，避免误把 LaTeX 字符串写入 PDF。
4. **Step 8D（已验收）：Reflow 矢量绘制与 QA。**
   - 先按固定文本 bbox 清除原文字层，再按公式自己的源 bbox 清除旧公式文字/路径；不扩大任何 bbox。
   - 原 PDF 公式使用 `show_pdf_page` 回放经哈希验证的矢量 clip；MathJax SVG 在内存中转换为单页矢量 PDF 后用同一接口绘制，不经过位图。
   - 文字 run 和公式 run 共享 Step 8C 的行基线；公式目标框直接使用已测量的 point 宽高和行内偏移。
   - 输出 `reflow-feedback-v2`，逐元素及全文校验公式数量、顺序、固定 bbox、基线误差、公式间重叠、目标区域非空和矢量绘制路径。
   - 任何公式资产哈希变化、缺失、越界、顺序错误、空白绘制或非矢量路径都会使本次输出失败；失败产物不会保留。

Step 8D 已完成公式矢量绘制和结构化 QA；生产级验收仍需在 Step 9 接入真实 API 译文后执行。

### 10.2 真实 Provider 与端到端验收（Step 9）

Step 9 的代码路径已实现，但付费请求需要本地环境先配置凭据：

- `hammer-translator --run` 是唯一真实执行入口（流水线 CLI），会明确检查输出目录和 Provider 凭据。兼容入口 `translate_pdfs.py --run` 仍可使用。
- 通过 OpenAI-compatible Chat Completions 调用 DeepSeek/OpenAI；密钥从进程环境或项目根目录 `.env` 读取，已存在的进程环境变量优先。
- `.env` 仅对当前用户可读写并被 `.gitignore` 排除；`.env.example` 只保留空值模板。可用 `--env-file` 显式选择其他文件。
- 远程错误转换为可恢复/不可恢复类型，不在日志中写入远程响应体或密钥。
- 初译和固定 bbox 受约束重译共用同一 Provider，并继续使用缓存、checkpoint 和保护内容往返校验。
- 执行报告记录实际网络请求数和 token 用量，不根据可变的价目表猜测费用。
- 默认向 stderr 实时报告单元进度、API 尝试/重试和布局阶段；通过 Conda 启动时需用 `conda run --no-capture-output` 禁用 Conda 的输出缓存。
- 中断后可沿用 checkpoint 继续；如果已有单元标记为 `failed`，显式使用 `--retry-failed` 重试，避免静默跳过。
- 离线回归已覆盖“真实 Provider 代码路径 → `translation-v3` → 薄 Reflow → PDF”；最终验收尚需真实译文及逐页视觉检查。

标准命令行入口（推荐 `hammer-reflow`，兼容旧脚本 `render_translated_pdf.py`）：

```bash
conda run -n hammertrans hammer-reflow \
  path/to/document_translated.json \
  --output path/to/document_translated.pdf \
  --feedback path/to/document_reflow_feedback.json
```

只有在需要导出一份保留未解决原文的审阅预览时，才增加 `--allow-partial`。

下一步的总开发范围是：**重写 Translation Stage + 抽取 Translation/Reflow 共享的 LayoutMetrics + 实现薄 Reflow 兼容层**，而不是立即重写整个 Reflow。

## 11. 分步实施与审批规则

后续重写必须严格分步进行。每个步骤完成后，先汇报以下内容并等待用户认可，不自动开始下一步：

1. 本步实际修改的文件和边界。
2. 已完成的功能与明确未包含的功能。
3. 测试项、测试结果和未覆盖风险。
4. 数据契约或行为与上一步相比的变化。
5. 下一步拟实施的范围，由用户明确同意后继续。

建议检查点：

- **Step 1（已验收）：安全和唯一入口。** 密钥外置、新 CLI 骨架、preflight/dry-run，不调用真实 API。标准入口为 `hammer-translator`（流水线 CLI）；旧入口在发起 API 请求前停止。
- **Step 2（已验收）：输入 Schema 和 Loader。** 已固定 enriched/content-list 发现规则，实现 `translation-input-v1` 验证器、稳定 `block_id`、PDF 页码/bbox 校验和统一 taxonomy；原始块及其扩展字段不被改写。
- **Step 3（已验收）：翻译单元和内容保护。** 已实现逻辑段落聚合、表格主体与 caption 分离、`image_caption` 归类、作者/机构保守拆分，以及公式/URL/DOI/引用/代码/HTML/数值占位符的严格往返校验。
- **Step 4（已验收）：Provider、缓存和断点恢复。** 已实现 Provider 中性请求/响应、本地 `FakeProvider`、SQLite 内容哈希缓存、每文档原子 checkpoint、错误分类/退避重试与 `--simulate` 本地执行入口；未调用真实 API。
- **Step 5（已验收）：初译、QA 和版本历史。** 已实现 `translation-v1` 可读产物、`initial_text/candidates/selected_text`、翻译状态、初译质量报告、block-to-unit 映射和原子写入；`FakeProvider` 结果明确标记为 `simulated`，不伪装成真实译文。
- **Step 6（已验收）：黄金样式与共享 LayoutMetrics。** 已实现原始 PDF 样式采样、分层稳健统计、中文字体映射、黄金参数边界、`layout-style-v1` 和共享 `layout-metrics-v1`；bbox 始终只读。
- **Step 7（已验收）：Translation 译文切分与受约束重译。** 已实现多 bbox 连续切分、全部可行候选评分、排版预检、不可拆内容边界、受约束重译与缓存，并输出逐元素 `translated/recommended_style/layout_preflight`。
- **Step 8（已完成）：Reflow 兼容层。** 已实现产物和字体哈希校验、共享测量复核、固定 bbox 文字替换、部分预览、字体子集化、PDF 几何/文字层验证和 `reflow-feedback-v1`；Reflow 本身不改写译文、不调用 LLM。
- **Step 8A（已验收）：公式清单与契约。** 已实现 MinerU 公式 span 提取、独立 bbox/来源保留、公式占位符绑定、稳定 ID、几何异常标记和 `formula-contract-v1`。
- **Step 8B（已验收）：公式矢量资产引擎。** 已实现受哈希保护的原 PDF clip、clip 污染检测/缩紧、MathJax SVG 降级、资产安全校验、基线指标和 `formula-assets-v1`。
- **Step 8C（已验收）：`LayoutMetrics v2` 混合 run 测量。** 已实现公式原子 run、真实矢量尺寸缩放、共享基线、混合换行、多 bbox 归属更新、Translation/Reflow 复算一致性与 v1 兼容。
- **Step 8D（已验收）：Reflow 矢量公式绘制与视觉/结构 QA。** 已实现原 PDF clip 和 MathJax SVG 的矢量回填、旧公式定向清除、共享基线绘制，以及公式数量、顺序、bbox、基线、重叠、非空与矢量路径验证。
- **Step 9（进行中）：真实 API 与端到端 PDF 验收。** 真实 Provider、`--run`、用量记录和离线端到端测试已完成；待配置 `TRANSLATION_API_KEY` 后执行付费译文与最终视觉验收。
