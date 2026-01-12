# LLM2SQLStructuredSearch

**LLM2SQLStructuredSearch** is a proof-of-concept (POC) project that explores how to improve the **controllability, correctness, and interpretability** of Text-to-SQL generation by combining:

* **Structured schema search**
* **Two-phase retrieval (table → column/join)**
* **Programmatic validation and iterative correction loops**

The project is built on the **MiniDev dataset** and compares two approaches:

* **Baseline (Old Approach)** : Directly prompting an LLM to generate SQL in a single pass
* **Two-Phase Structured Search (New Approach)** :

  *Sketch → Search → Validate → Minimal Edit Refinement → (Optional Verification)*

---

## 0. One-Sentence Goal

Given a natural language question (NLQ), generate **high-quality, schema-valid SQL** by:

1. Producing multiple SQL sketches instead of a single final query
2. Aligning sketches with database schema via structured retrieval
3. Enforcing join/path validity using hard constraints
4. Iteratively refining SQL through minimal edits and validation feedback
5. Producing fully traceable intermediate artifacts for debugging and analysis

---

## 1. Motivation

End-to-end Text-to-SQL generation often fails in realistic databases due to:

* Large schemas with many tables and columns
* Ambiguous or overlapping column semantics
* Complex multi-hop JOIN paths and foreign key relationships
* Lack of hard constraints and verification in one-shot generation

As a result, LLMs frequently hallucinate tables/columns, produce invalid joins, or generate SQL that is syntactically correct but semantically wrong.

This project proposes a **structured search + validation-enhanced pipeline** that shifts LLMs away from “one-shot SQL generation” toward  **controlled, iterative convergence** .

---

## 2. Core Idea: Two-Phase Structured Search Pipeline

### 2.1 Sketch Generation (Candidate SQLs)

Instead of generating a single SQL query, the LLM first produces  **multiple SQL sketches** :

* SQL templates may contain **unknown constants or placeholders** (e.g. `:year`, `:start_date`)
* Different candidates explore alternative joins, aggregations, and filters
* Generation is intentionally **diverse** (non-homogeneous)
* Output is required to be **strict JSON** for downstream automation

The goal is to capture intent diversity early and reduce the risk of committing to a wrong structure.

---

### 2.2 Entity Extraction (Dual-Channel)

From each SQL sketch, structural entities are extracted via two channels:

* **LLM-based extraction**
  * Tables used
  * Columns used (table, column)
  * Join paths
* **Parser-based extraction**
  * Using **sqlglot** (or a pluggable SQL parser)
  * Programmatic extraction of the same entities

Discrepancies between the two channels are recorded as **warning signals** and later used as penalties during scoring.

---

### 2.3 Schema Indexing (Embedding + Metadata)

The database schema is indexed as a searchable structure:

* Embeddings for:
  * Table names
  * Column names
  * Foreign-key / relationship edges
* Additional metadata (recommended / extensible):
  * Synonyms and aliases
  * Column data types
  * Example values
  * Table row counts and column cardinalities
* A **Join Graph** is constructed to represent valid join paths

This schema index serves as the backbone for structured retrieval.

---

### 2.4 Two-Phase Retrieval and Scoring

For each SQL sketch, structured alignment is performed in two stages:

#### Phase 1: Table Retrieval

* Retrieve Top-K candidate tables relevant to the sketch intent

#### Phase 2: Column & Join Alignment

* Retrieve Top-K columns within candidate tables
* Compute valid join paths using the Join Graph
* Generate a set of **allowed_joins** as hard constraints

Each sketch is associated with an  **evidence package** , including:

* Matched tables, columns, and join paths
* Similarity scores
* Replacement suggestions for invalid entities

A unified score is computed, for example:

<pre class="overflow-visible! px-0!" data-start="4167" data-end="4285"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>Score </span><span>=</span><span></span><span>0.4</span><span> · </span><span>avg</span><span>(table_score)
      </span><span>+</span><span></span><span>0.4</span><span> · </span><span>avg</span><span>(column_score)
      </span><span>+</span><span></span><span>0.2</span><span> · </span><span>avg</span><span>(join_score)
      − penalties
</span></span></code></div></div></pre>

 **Penalty examples** :

* Non-existent tables or columns
* Disconnected joins
* Join paths not in `allowed_joins`
* `SELECT *` usage
* Aggregations without proper `GROUP BY`

---

### 2.5 Minimal-Edit Refinement with LLM

The LLM is then asked to **minimally edit** the SQL sketch using:

* Original sketch and extracted entities
* Evidence package (Top-K matches)
* `allowed_joins`
* Validation violations (see verification layer)

Rules:

* Only invalid or uncertain parts may be modified
* Prefer Top-1 / Top-2 retrieved entities
* Output must be  **strict JSON** , including:
  * Refined SQL
  * `confidence`
  * `delta_notes` (what changed and why)

---

### 2.6 Iterative Correction Loop

After each refinement:

* Programmatic validation is executed
* Scores and confidence are updated
* Attempt history is recorded to avoid repeated bad paths

Stopping conditions:

* Schema-valid SQL with score ≥ threshold (e.g. 0.75 / 0.8)
* Maximum iterations reached (typically  **2–3** )
* No improvement over two consecutive iterations → early stop with best candidate

If all attempts fail, the system escalates to  **human review** , providing full evidence and alternatives.

---

## 3. Optional Verification Layers

To further improve reliability, optional verification layers can be enabled:

* **L0 – Static Validation**
  * Tables/columns ∈ schema whitelist
  * All joins ∈ `allowed_joins`
  * No `SELECT *`
  * Proper `GROUP BY` for aggregations
* **L1 – Lightweight Execution**
  * `EXPLAIN` or `LIMIT` dry-run
  * Timeout protection
  * Syntax/object existence checks
* **L2 – Semantic Sanity Checks**
  * Type and aggregation shape validation
  * Top-N / min / max intent alignment
  * Fact vs dimension table consistency
* **L3 – Sampling Verification (Optional)**
  * Recompute aggregates via subqueries on sampled results

Verification results feed back into the correction loop.

Failures are packaged with evidence and routed to human reviewers.

---

## 4. Project Structure

<pre class="overflow-visible! px-0!" data-start="6280" data-end="8358"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>LLM2SQLStructuredSearch/
├── backend/
│   └── src/
│       ├── baseline/                 # Baseline: direct NL → SQL
│       ├── config/
│       │   └── config.py
│       ├── data_io/
│       │   ├── file_reader.py
│       │   └── file_writer.py
│       ├── embedding/
│       │   ├── embedding_client.py
│       │   ├── column_names_embedder.py
│       │   ├── edge_names_embedder.py
│       │   └── output/
│       │       ├── columns_name_embeddings.json
│       │       └── edges_name_embeddings.json
│       ├── evaluator/
│       │   ├── sql_refiner.py
│       │   └── batch_sql_refiner.py
│       ├── llm/
│       │   └── chatgpt_client.py
│       ├── prompt/
│       │   ├── system_sql_candidates.txt
│       │   ├── user_sql_candidates_template.txt
│       │   ├── system_sql_refine.txt
│       │   └── user_sql_refine_template.txt
│       ├── query/
│       │   ├── sql_candidate_generator.py
│       │   ├── batch_sql_candidate_generator.py
│       │   ├── sql_structure_embedding_builder.py
│       │   ├── batch_sql_structure_embedding_builder.py
│       │   └── output/
│       │       ├── sql_candidates_0001.json
│       │       ├── sql_candidates_0002.json
│       │       ├── sql_candidates_0003.json
│       │       ├── sql_structure_embeddings.json
│       │       └── batch/
│       │           ├── *_sql_refinement_*.json
│       │           └── *_structure_embedding_*.json
│       ├── schema_parser/
│       ├── twophase/
│       │   ├── sql_candidate_searcher.py
│       │   └── batch_sql_candidate_searcher.py
│       └── utils/
│           └── utils.py
├── data/
│   └── raw/
│       └── data_minidev/
│           └── MINIDEV/
│               ├── dev_databases/
│               ├── dev_tables.json
│               ├── mini_dev_mysql.json
│               ├── mini_dev_mysql_gold.sql
│               ├── mini_dev_postgresql.json
│               ├── mini_dev_postgresql_gold.sql
│               ├── mini_dev_sqlite.json
│               └── mini_dev_sqlite_gold.sql
├── tests/
├── main.py
├── README_cn.md
├── README_en.md
└── requirements.txt
</span></span></code></div></div></pre>

---

## 5. Outputs and Traceability

All intermediate artifacts are persisted for reproducibility:

* Schema embeddings (tables / columns / edges)
* SQL candidates per batch
* Refined SQL versions
* SQL structure embeddings
* Retrieval evidence and scoring (extensible)

This design enables debugging, A/B testing, and research analysis.

---

## 6. Dataset References

* **MiniDev Dataset**

  Lightweight cross-database Text-to-SQL dataset

  [https://github.com/bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
* **Spider 1.0**

  [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)
* **Spider 2.0**

  [https://github.com/xlang-ai/Spider2](https://github.com/xlang-ai/Spider2)

MiniDev is used for rapid experimentation; Spider datasets are future expansion targets.

---

## 7. Baseline vs Two-Phase Approach

**Baseline**

* One-shot SQL generation from NL
* No hard constraints or validation
* Difficult to debug or correct

**Two-Phase Structured Search**

* LLM focuses on intent sketches
* Retrieval + rules act as hard guardrails
* Iterative, explainable correction
* Verification-backed escalation to human review

---

## 8. Roadmap (Optional)

* Richer SQL AST / logical plan embeddings
* Execution-guided feedback loops
* Stronger Phase-2 re-rankers (rules + LLM judges)
* Standardized evaluation metrics (EM / execution accuracy)
