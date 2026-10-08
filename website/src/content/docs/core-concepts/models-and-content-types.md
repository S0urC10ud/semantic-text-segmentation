---
title: Models & Content Types
description: Choose U-Net for throughput or Mamba for context, across 35 learned textual labels.
---
## Choose a model

| Model | Python API | Good starting point for | Parameters |
| --- | --- | --- | --- |
| U-Net | `typeseg.fast(text)` | CPU deployment and bulk text inspection | 1.52M |
| Mamba | `typeseg.precise(text)` | More context and finer boundary localization | 1.36M |

U-Net uses overlapping windows; Mamba carries context across the file. Start with U-Net when throughput matters. Compare Mamba on representative inputs if boundary quality is more important than speed. See [Performance & Evaluation](../../security/performance/) for measured results.

Both models in the public package are **general-domain checkpoints**. The paper separately evaluates U-Net-Sec and Mamba-Sec after security-domain adaptation; those results are not measurements of the bundled package models.

## Supported content types

The model learns 35 textual labels:

| Category | Labels |
| --- | --- |
| Programming languages (17) | `python`, `javascript_typescript`, `java`, `c_family`, `csharp`, `go`, `rust`, `ruby`, `php`, `swift`, `kotlin`, `scala`, `dart`, `visual_basic`, `shell`, `powershell`, `sql` |
| Markup and documentation (7) | `html`, `xml`, `svg`, `css`, `markdown`, `restructuredtext`, `tex` |
| Data and configuration (6) | `json`, `yaml`, `csv`, `text`, `dockerfile`, `gettext_catalog` |
| Text encodings (5) | `encoding_hex`, `encoding_base64`, `encoding_base32`, `encoding_base58`, `encoding_base85` |

`c_family` groups C, C++, and Objective-C. `javascript_typescript` groups JavaScript and TypeScript. The browser uses the shorter display name `js_ts`.

Confidence gating adds an auxiliary `other` output for uncertain positions. It is not a learned 36th class or a guarantee that all unsupported languages will be recognized as unknown.

TypeSeg handles textual content. Binary executables, archives, images, and overlapping interpretations of the same bytes are outside its current scope.
