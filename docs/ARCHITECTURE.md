# Architecture Guide

EviCode separates verification into evidence extraction, evidence fusion, and evidence-grounded explanation.

```text
translation pair
  -> evidence extraction
      -> syntax evidence
      -> AST evidence
      -> CFG evidence
      -> API evidence
      -> identifier evidence
      -> execution evidence
      -> retrieval evidence
  -> evidence vector
  -> fusion model
  -> verification score
  -> verification report
```

Each evidence source implements `extract()`, `score()`, and `explain()`.

The first implemented smoke modules are intentionally lightweight. Rich AST, CFG, API, identifier, retrieval, and execution modules are scheduled in `progress/todo.json`.

The scientific contribution is not only the final score. The framework must quantify:

- which static evidence sources most closely approximate execution,
- which sources are complementary,
- how each source's marginal value changes with execution budget,
- which evidence failures explain likely semantic mismatch.
