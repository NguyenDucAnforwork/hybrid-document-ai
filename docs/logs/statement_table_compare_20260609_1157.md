# Statement Table Parser Comparison 20260609_1157

- Dataset: `C:\docai-demo-ws\data\statements_test_hard`
- N=8 hard statements (OCR tokens + parser only)
- Debug artifacts: `docs\logs\statement_table_debug_20260609_1157`

| mode | row-F1 | amount-acc | description-acc | mean latency (ms) | p50 latency (ms) | row reconcile | closing reconcile |
|---|---|---|---|---|---|---|---|
| rules | 0.943 | 0.413 | 0.987 | 5.1 | 3.0 | 0.347 | 0.0 |
| tatr | 0.791 | 0.291 | 0.873 | 3060.0 | 386.8 | 0.206 | 0.0 |
| hybrid | 0.943 | 0.4 | 0.92 | 392.9 | 390.6 | 0.357 | 0.0 |

## Interpretation

- `rules`: current heuristic row/column parser.
- `tatr`: zero-shot Table Transformer structure recognition only.
- `hybrid`: use `tatr` when structure assignment looks trustworthy, else fall back to `rules`.

## Debug

- rules: `docs\logs\statement_table_debug_20260609_1157\rules`
- tatr: `docs\logs\statement_table_debug_20260609_1157\tatr`
- hybrid: `docs\logs\statement_table_debug_20260609_1157\hybrid`