# Language-Normalized Evidence Redesign

This note records the Phase III redesign of EviCode from raw cross-language syntax comparison toward language-normalized source-to-candidate evidence.

## Core Position

EviCode should not claim that raw Python and Java syntax trees are directly semantically comparable. The redesigned extractor therefore keeps weak cross-language proxies visible, but shifts the primary static evidence toward properties extracted independently from each program and then compared as language-normalized profiles.

The current setting is **reference-free with respect to target-language gold implementations** for external LLM translations: evidence is computed between the source program and the generated candidate, not between the candidate and a gold Java reference. It is still source-conditioned verification, not generated-code-only validation.

## Evidence Classification

EviCode now reports evidence through four high-level families:

| Family | What It Measures | Examples |
|---|---|---|
| Structural | Program organization independent of syntax | loops, branches, nesting, CFG-size proxies, statement distribution, function/class structure |
| Behavioral | Operations and value movement | operator families, returns, exceptions, calls, identifier roles, variable usage, data-flow summaries |
| Complexity | Density and decision structure | cyclomatic complexity, decision density, expression density, assignment density, tree depth |
| Reliability | Whether evidence can be trusted or behavior was observed | parse validity, syntax validity, execution availability, execution pass/fail, confidence estimates |

| Evidence | Feature | Classification | Interpretation |
|---|---|---|---|
| Token overlap | `token_jaccard` | Weak cross-language proxy | Useful for cheap plausibility but sensitive to syntax and naming. |
| Edit similarity | `edit_similarity` | Weak cross-language proxy | Raw text similarity; not semantic across languages. |
| AST node similarity | `ast_similarity` | Weak cross-language proxy | Compares language-specific parser node names, so it is de-emphasized. |
| API overlap/mismatch | `api_similarity`, `api_mismatch_score` | Weak cross-language proxy | API names differ across languages; retained for diagnostic hints only. |
| Syntax validity | `ln_syntax_both_valid` | Language-normalized | Each program is parsed in its own language. |
| CFG size | `ln_cfg_nodes_similarity`, `ln_cfg_edges_similarity` | Language-normalized | Compares control-flow size proxies. |
| Control structure | `ln_control_profile_similarity` | Language-normalized | Compares branch, loop, return, and exception profiles. |
| Complexity | `ln_cyclomatic_complexity_similarity` | Language-normalized | Compares decision-count complexity. |
| Structural shape | `ln_max_ast_depth_similarity`, `ln_avg_tree_depth_similarity`, `ln_branching_factor_similarity` | Language-normalized | Uses tree-depth and branching statistics, not raw node labels. |
| Statement distribution | `ln_statement_distribution_similarity` | Language-normalized | Compares branches, loops, returns, assignments, functions, classes, and exceptions. |
| Expression distribution | `ln_expression_distribution_similarity` | Language-normalized | Compares call, operator, literal, indexing, and attribute-access statistics. |
| Operator families | `ln_operator_family_similarity` | Language-normalized | Compares arithmetic, comparison, logical, and assignment operator families. |
| Identifier statistics | `ln_identifier_count_similarity`, `ln_identifier_entropy_similarity` | Language-normalized | Compares identifier quantity and diversity. |
| Identifier-role distribution | `ln_identifier_role_distribution_similarity` | Language-normalized | Compares declaration, assignment, and call-role profiles. |
| Data-flow proxies | `ln_def_use_count_similarity`, `ln_read_write_ratio_similarity`, `ln_assignment_density_similarity` | Language-normalized | Compares approximate value movement and assignment structure. |
| Calls | `ln_call_count_similarity` | Language-normalized | Compares function/method call volume. |
| Execution | `execution_passed_example`, `execution_passed_full` | Dynamic evidence | Behavior observed under available tests. |

## Mathematical Form

For each program \(p\), EviCode extracts a language-normalized profile:

\[
\phi(p) = [c(p), s(p), o(p), i(p), d(p), q(p)]
\]

where \(c\) denotes control-flow statistics, \(s\) structural statistics, \(o\) operator-family statistics, \(i\) identifier statistics, \(d\) approximate data-flow statistics, and \(q\) validity/complexity statistics.

For scalar properties \(a\) and \(b\), similarity is:

\[
\operatorname{sim}(a,b)=\frac{\min(a,b)}{\max(a,b,1)}.
\]

For histogram properties \(u\) and \(v\), similarity is cosine similarity:

\[
\operatorname{sim}(u,v)=\frac{u\cdot v}{\|u\|\|v\|}.
\]

This means Python and Java programs are not compared by raw tree labels. They are independently mapped into comparable evidence vectors and then compared.

## Current Limitations

The profile extractor is still lightweight. CFG nodes, data-flow, and call graph features are approximations, not full compiler-grade analyses. The redesign strengthens the scientific framing, but future work should replace these proxies with parser-backed CFGs, def-use graphs, call graphs, and type-aware API normalization.
