# SZFY Investment Brain — WorkBuddy Task

## 0. Mission

Design and validate an independent **SZFY (市值风云 Investment Brain)** system.

This is **not** a summary database, article-card archive, or simple embedding/RAG application.

The goal is to distill reusable analytical methods, reasoning chains, decision frameworks, and author/editorial-persona styles from roughly **8,804 archived 市值风云 articles**, then reassemble them into a system that can analyze **new stocks, new sectors, new market states, and new investment questions**.

Target transformation:

```text
Articles
  -> Decisions / Reasoning
  -> Canonical Methods
  -> Decision Capabilities
  -> Current Facts
  -> New Analysis / Judgment
```

A successful system should eventually support queries such as:

```text
分析 000651 格力电器
格力是不是龙头？为什么？
格力现在便宜吗？
怎么判断一家公司是不是高质量公司？
筛选 A 股里的高质量龙头
现在半导体处于什么周期？
当前市场情绪怎么样？
当前市场主线是什么？
有哪些判断龙头的方法？按可靠性排列。
这个方法来自哪些市值风云研报？
不同作者对同一个问题用了哪些不同方法？
```

The answer must be **analysis + judgment**, not merely a list of retrieved articles.

---

# 1. Core architecture principle

The system must explicitly separate:

## METHOD_PROVENANCE
Historical SZFY articles primarily teach:

> **HOW TO ANALYZE**

Examples:
- how to identify a leader;
- how to judge an industry cycle;
- how to detect false prosperity;
- how to validate earnings quality;
- how to judge valuation;
- how to identify a market mainline;
- how to reason about capex and supply/demand.

## FACT_PROVENANCE
Current data tells the system:

> **WHAT IS TRUE NOW**

Examples:
- revenue, profit, CFO;
- inventory, receivables;
- ROIC;
- product prices;
- capex;
- valuation;
- market trend, relative strength;
- ETF/capital flows;
- breadth, turnover, sentiment.

The allowed chain is:

```text
SZFY Method
    +
Current Facts
    -> Reasoning Instance
    -> Current Judgment
```

Forbidden:

```text
old article conclusion
    -> pretend it is a current fact or current judgment
```

Historical conclusions may remain evidence, but cannot become current facts without current verification.

---

# 2. Three adjudicated decisions

## 2.1 Author axis: dual identity, no guessing

Use two parallel identity layers.

### A. AuthorIdentity
A real individual/team author only when supported by evidence such as:
- metadata;
- article signature;
- author page;
- stable reliable attribution.

If unsupported, use `UNKNOWN`.
Do not guess.

### B. EditorialPersona / SeriesPersona
Even when individual authors are unavailable, preserve stable editorial/series identities such as recurring columns or series.

These may have their own:
- covered domains;
- preferred evidence;
- frequently used Methods;
- decision style;
- historical evolution.

Low author coverage must not collapse the author/brain layer.
Pilot must report actual attribution coverage, without a pre-fixed threshold such as 30%.

## 2.2 Pilot sample size

Use **60–80 targeted articles**, not a random sample.

Deliberately preserve enough overlap to test whether candidate Methods truly converge.

Cover:
- multiple identifiable authors when available;
- multiple Editorial/Series Personas;
- multiple industries;
- multiple years / market regimes where practical;
- long, short, and image-heavy articles;
- series and non-series;
- market, sector, stock, valuation, cycle, flow, governance, risk and falsification decisions.

Targeted overlap is encouraged.
Do not manufacture weak Method merges merely to hit a count.

## 2.3 Method granularity

Canonical merge basis = **mechanism + required_inputs**.

Rule:

> If two candidate methods have substantively the same mechanism and materially equivalent required inputs, but differ only in target / industry / example / wording, treat them as **variants of one Canonical Method**.

> If the underlying mechanism differs, preserve them as **different Methods**, even if they solve the same Problem. They may be `complements`, `conflicts_with`, or `alternative_to`.

Do not merge based only on title, topic, or keywords.

---

# 3. Core domain objects

Design and validate at least:

- `Article`
- `AuthorIdentity`
- `EditorialPersona` / `SeriesPersona`
- `Decision`
- `Evidence`
- `Reasoning`
- `Method`
- `Problem`
- `DecisionCapability`
- `FactRequirement`
- `CurrentFact`
- `Judgment`
- `CounterEvidence`
- `Outcome`

Recommended relationships:

```text
Article CONTAINS Decision
Decision SUPPORTED_BY Reasoning
Reasoning INSTANTIATES Method
Method SOLVES Problem
Method REQUIRES FactRequirement
AuthorIdentity USES Method
EditorialPersona USES Method
Method SUPPORTED_BY Article
DecisionCapability ORCHESTRATES multiple Methods
Judgment APPLIES Method
Judgment USES CurrentFact
Judgment CHALLENGED_BY CounterEvidence
Judgment TRACEABLE_TO Article
```

The core reusable asset is not the Article.
It is **Canonical Method + DecisionCapability**.

---

# 4. Decision object and reasoning chain

One article may contain multiple Decision Objects.
Do not force one article -> one conclusion.

Each high-value Decision should support at least:

- `decision_question`
- `decision_type`
- `target`
- `decision`
- `horizon`
- `evidence`
- `reasoning_chain`
- `method_used`
- `assumptions`
- `source_failure_condition`
- `derived_boundary`
- `counter_evidence`
- `confidence/evidence_strength`
- future PIT-safe `outcome`

Reasoning should aim to preserve:

```text
facts
-> intermediate variables
-> mechanism
-> comparison / exclusion
-> key inference
-> decision
```

The system must not retain only the final conclusion.

---

# 5. Canonical Method object

Each Method should include at least:

- `method_id`
- `method_name`
- `problem_types`
- `decision_capabilities`
- `required_inputs`
- `procedure`
- `mechanism`
- `outputs`
- `applicable_scope`
- `not_applicable_scope`
- `source_failure_conditions`
- `derived_boundaries`
- `common_misuse`
- `counterexamples`
- `source_articles`
- `source_authors`
- `source_editorial_personas`
- `variants`
- `depends_on`
- `complements`
- `conflicts_with`
- `cross_article_repeat_count`
- `cross_author/persona_support`
- `cross_industry_transferability`
- future validation metadata

---

# 6. Decision Capability layer

DecisionCapability must be a first-class structure above Methods.

At minimum consider:

## MARKET
- market state / direction
- sentiment
- risk appetite
- breadth
- capital / ETF flow
- mainline identification
- mainline persistence

## INDUSTRY / SECTOR
- sector attractiveness
- cycle position
- supply/demand
- inventory
- product prices
- capacity
- capex cycle
- policy
- competitive structure
- profit-pool migration

## STOCK
- quality
- value
- growth
- leader
- competitive advantage / moat
- business model
- ROIC / capital efficiency
- cash-flow quality
- management / governance
- catalyst
- turnaround
- cyclical stock analysis
- risk / falsification / red flags

## VALUATION
- PE / PB / EV-EBITDA
- FCF
- ROIC economics
- normalized-cycle earnings
- SOTP
- peer relative valuation
- historical range
- implied growth assumptions

## SCREENING
- quality screen
- value screen
- leader screen
- cycle screen
- reversal screen
- composable multi-method screens

---

# 7. Multi-method analysis is required

One problem may need several Methods in parallel.

Example: “Is Gree cheap?” might invoke:
- historical valuation;
- peer relative valuation;
- normalized-cycle earnings;
- FCF / cash generation;
- ROIC / reinvestment economics;
- implied-growth analysis.

The system should expose:

```text
Methods considered
Methods selected
Why selected
Method outputs
Agreement / disagreement
Unknowns
Dominant conclusion
```

Until calibration is validated, do not fabricate precise probabilities or pseudo-scientific scores.
Prefer truthful ordinal outputs such as:

`STRONG / MODERATE / WEAK / UNKNOWN`

---

# 8. Query orchestration

Do not design this as:

```text
stock code
-> embedding search articles
-> LLM summary
```

That is only RAG.

Design an explicit orchestration pipeline, for example:

```text
USER QUERY
  -> Query Resolver
  -> Problem Classifier
  -> DecisionCapability Selector
  -> Method Retriever / Ranker
  -> Fact Requirement Resolver
  -> Current Facts
  -> Method Execution / Reasoning
  -> Counter-evidence Search
  -> Multi-method Synthesis
  -> Judgment
  -> Provenance
  -> Immutable Analysis Run
```

RAG may exist inside retrieval, but RAG is not the architecture.

---

# 9. Method ranking / selection

For the same Problem, design a Method Retriever + Ranker using factors such as:

- `problem_fit`
- `applicability`
- `required_data_availability`
- `source_support`
- `cross_article_repeat`
- `cross_author/persona_support`
- `cross_industry_transferability`
- future `historical_validation`
- `conflict_level`
- `freshness relevance` where applicable

The first version may use rule + judgment logic.
Do not invent fixed weights merely to make the design look complete.

---

# 10. Provenance contract

Every material current Judgment must support a trace like:

```text
Judgment
  -> Reasoning Instance
      -> Method
          -> Source Articles
          -> Author / Persona
      -> Current Facts
          -> Fact Provider
          -> as_of / known_at
  -> CounterEvidence
```

For each SZFY article source preserve at least:
- `article_id`
- title
- date
- author/persona when supported
- source id/path/url
- reliable section/excerpt pointer if available
- extraction provenance

If exact source location is unavailable, say `UNKNOWN` rather than fabricating precision.

---

# 11. User-facing output

Use the principle:

`FULL TRACE IN BACKEND / COMPACT JUDGMENT TO USER`

Example default stock output:

```text
SZFY | 000651 格力电器

Overall Judgment: ...
Quality: STRONG
Value: MODERATE
Growth: WEAK-MODERATE
Leader: STRONG
Cycle: NEUTRAL
Market Support: UNKNOWN / MODERATE
Risk: MODERATE

Dominant Judgment:
...

Top Methods:
M-...
M-...
M-...

Main Counter-evidence:
...

Source Support:
12 SZFY articles / 3 authors / 2 series personas
```

Then allow detail-lazy expansion:

```text
why
methods
sources
reasoning
counterevidence
authors
facts
```

Every material displayed field should map to a persisted Analysis Run rather than be regenerated from memory.

---

# 12. Relationship with WTOS

SZFY is initially:

`Research / Method / Judgment Brain`

WTOS remains:

`Unified Decision / Risk / Execution System`

Hard invariant:

`SZFY_JUDGMENT != WTOS_AUTHORIZATION`

Do not modify any private WTOS/Stock-Analysis production runtime, authority, workflow, trading rule, or broker behavior.

You may inspect **public GitHub repositories** for architectural patterns relevant to:
- knowledge graphs;
- research provenance;
- retrieval + reasoning systems;
- investment research tools;
- agent orchestration;
- immutable run records;
- evaluation / replay / audit.

Use public GitHub only as **reference evidence**, not as authority. Cite which public repositories/patterns materially influenced the design, and explain what was borrowed vs rejected.

---

# 13. Pilot v2 acceptance tests

The 60–80 article Pilot must test whether the distilled brain can answer actual decision questions, not merely whether extraction works.

At minimum test:

1. 怎么判断一家公司是不是龙头？
2. 怎么判断一家公司是不是高质量公司？
3. 怎么判断一只股票便宜不便宜？
4. 怎么判断行业进入上行周期？
5. 怎么判断热门板块是真产业趋势还是资金炒作？
6. 怎么判断利润增长但质量恶化？
7. 同一个问题有哪些不同 Method？
8. 这个 Method 来自哪些文章 / 作者 / 栏目？
9. 同一作者 / 栏目长期反复用了什么方法？
10. 不同作者 / 栏目对同一问题用了哪些不同 mechanism？

If the Pilot can only answer “what did article X say?”, treat the design as failed.

---

# 14. Deliverables

Do not return to ask routine design questions. Execute this round and return one consolidated result.

At minimum deliver:

1. `00_EXECUTIVE_DESIGN.md`
2. `01_FIRST_PRINCIPLES.md`
3. `02_SYSTEM_ARCHITECTURE.md`
4. `03_DOMAIN_OBJECT_MODEL.md`
5. `04_CLASSIFICATION_AND_AUTHOR_PERSONA_SCHEMA.md`
6. `05_METHOD_TAXONOMY_AND_GRANULARITY.md`
7. `06_DECISION_CAPABILITY_MAP.md`
8. `07_QUERY_CONTRACT.md`
9. `08_OUTPUT_CONTRACT.md`
10. `09_PROVENANCE_CONTRACT.md`
11. `10_SZFY_WTOS_INTERFACE.md`
12. `11_PILOT_V2_PLAN_AND_RESULTS.md`
13. schema drafts for Decision / Reasoning / Method / Judgment etc.
14. actual 60–80 article Pilot extraction results
15. Method clustering results
16. author/persona attribution coverage
17. credible multi-article -> same Method merge cases when supported
18. same Problem -> different Method cases when supported
19. conceptual acceptance-test results
20. MVP implementation plan
21. brief survey of relevant **public GitHub repositories/patterns** and what should / should not be reused

---

# 15. Boundaries

This round may:
- design the independent SZFY system;
- modify the local SZFY / SZFY-analysis project files;
- execute the 60–80 article Pilot;
- generate schemas, Method library, capability map and analysis examples;
- research relevant public GitHub repositories for architecture ideas.

This round must NOT:
- run all 8,804 articles;
- alter private WTOS / Stock-Analysis production systems;
- create any trade order;
- treat historical article conclusions as current facts;
- force-fill UNKNOWNs;
- use one giant prompt as a substitute for architecture;
- call simple RAG an Investment Brain;
- invent unvalidated fixed weights or precise scores.

---

# 16. Final target

> Do not merely organize 8,804 SZFY articles.
> Distill their reusable decision-making ability and reassemble it into an investment brain that can analyze today’s new stock, sector and market questions — with every important reasoning step traceable to specific source articles and methods.

When this round is complete, send one **consolidated return** containing:
- executive conclusion;
- key architecture decisions;
- Pilot data/results;
- canonical Method findings;
- major failures / UNKNOWNs;
- public GitHub references that materially changed the design;
- only the small number of remaining questions that truly require user adjudication.

Do not send repeated progress emails.