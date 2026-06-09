# Statement Table Parser Comparison 20260609_1213

- Dataset: `C:\docai-demo-ws\data\statements_test_hard`
- N=8 hard statements (OCR tokens + parser only)
- Debug artifacts: `docs\logs\statement_table_debug_20260609_1213`

| mode | row-F1 | amount-acc | description-acc | mean latency (ms) | p50 latency (ms) | row reconcile | closing reconcile |
|---|---|---|---|---|---|---|---|
| rules | 0.943 | 0.413 | 0.987 | 2.4 | 1.8 | 0.347 | 0.0 |
| tatr | 0.791 | 0.291 | 0.873 | 411.6 | 416.2 | 0.206 | 0.0 |
| hybrid | 0.943 | 0.413 | 0.96 | 376.4 | 377.8 | 0.347 | 0.0 |

## Interpretation

- `rules`: current heuristic row/column parser.
- `tatr`: zero-shot Table Transformer structure recognition only.
- `hybrid`: use `tatr` when structure assignment looks trustworthy, else fall back to `rules`.

## Debug

- rules: `docs\logs\statement_table_debug_20260609_1213\rules`
- tatr: `docs\logs\statement_table_debug_20260609_1213\tatr`
- hybrid: `docs\logs\statement_table_debug_20260609_1213\hybrid`