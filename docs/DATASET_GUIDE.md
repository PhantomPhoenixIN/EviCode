# Dataset Guide

The current paper uses CodeNetTrans-QS scored generator outputs.

Tracked input files live under:

```text
datasets/Predictions_by_LLMs/
```

The current controlled analysis retains translations from C, C++, C#, Python, Ruby, Kotlin, and Swift to Java. Java is fixed as the target language to keep target parsing, wrapper conventions, runtime assumptions, and API environment constant while source language and generator vary.

The current pipeline reads complete generator outputs and their released execution grades. It does not create synthetic negatives for the current paper.

The active source-language configuration is:

```text
authentic_only_study/config.yaml
```

Feature extraction excludes target references, execution outcomes, compiler outcomes, test results, generator identity, and problem identifiers from the inference feature matrix.
