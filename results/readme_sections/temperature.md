**chrF by temperature**

| Model | chrF @T=0.0 | chrF @T=0.3 | chrF @T=0.7 | chrF @T=1.0 | chrF @T=1.5 |
|---|---|---|---|---|---|
| Transformer (scratch) | 2.6 | 4.7 | 6.7 | 10.4 | 11.2 |
| mT5-small (fine-tuned) | 4.9 | 7.8 | 9.0 | 9.6 | 10.2 |

**Diversity (100 − self-BLEU between two sampled runs)**

| Model | div @T=0.0 | div @T=0.3 | div @T=0.7 | div @T=1.0 | div @T=1.5 |
|---|---|---|---|---|---|
| Transformer (scratch) | 0.0 | 87.6 | 99.7 | 99.8 | 99.9 |
| mT5-small (fine-tuned) | 0.0 | 69.0 | 66.3 | 73.6 | 97.4 |