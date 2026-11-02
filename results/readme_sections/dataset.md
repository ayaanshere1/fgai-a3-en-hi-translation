| Split | Pairs | Avg EN words | Avg HI words | EN vocab (types) | HI vocab (types) |
|---|---|---|---|---|---|
| Train | 1500 | 14.9 | 15.8 | 7159 | 7036 |
| Validation | 100 | 13.9 | 14.8 | 774 | 839 |
| Test | 60 | 14.4 | 15.6 | 533 | 576 |

**Filtering pipeline (applied to the 1.66M-pair training corpus):**

| step | pairs remaining |
|---|---|
| raw pairs | 1659083 |
| non-empty | 1652981 |
| 3-40 EN words, 3-50 HI words | 1097623 |
| no software artefacts | 1088413 |
| script check (HI Devanagari, EN Latin) | 1066123 |
| length ratio 0.5-2.5 | 1056981 |
| de-duplicated | 730729 |