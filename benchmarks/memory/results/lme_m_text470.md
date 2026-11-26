
#### LongMemEval-M original, all 470, text only, labels=paper

| unit | key | mode | n | R@5 | N@5 | R@10 | R@20 | N@10 | any@5 | frac@10 | sR@5 | sR@10 | sR@20 | sN@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round | user | ref-bm25 | 470 | 0.426 | 0.361 | 0.489 ±0.045 | 0.555 | 0.384 | 0.511 | 0.532 | 0.449 | 0.517 | 0.585 | 0.387 |
| round | user | bm25-plain | 470 | 0.557 | 0.490 | 0.645 ±0.043 | 0.704 | 0.516 | 0.655 | 0.691 | 0.602 | 0.681 | 0.726 | 0.522 |
| round | user | bm25 | 470 | 0.596 | 0.521 | 0.687 ±0.042 | 0.751 | 0.552 | 0.700 | 0.753 | 0.640 | 0.723 | 0.794 | 0.557 |
| session | user | ref-bm25 | 470 | 0.615 | 0.534 | 0.694 ±0.042 | 0.757 | 0.558 | 0.664 | 0.731 | – | – | – | – |
| session | user | bm25-plain | 470 | 0.728 | 0.652 | 0.796 ±0.036 | 0.838 | 0.673 | 0.783 | 0.845 | – | – | – | – |
| session | user | bm25 | 470 | 0.768 | 0.680 | 0.845 ±0.033 | 0.891 | 0.701 | 0.811 | 0.884 | – | – | – | – |

#### Hinny timings (one haystack per question; median of 5 runs per query)

| unit | key | haystacks | items/haystack (median) | ingest ms p50 | ingest ms p99 | mode | query ms p50 | query ms p99 | cold p50 | cold p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| round | user (text only) | 470 | 2431 | 26.28 | 53.26 | bm25 | 0.093 | 0.255 | 0.158 | 0.427 |
| session | user (text only) | 470 | 502 | 26.11 | 46.79 | bm25 | 0.077 | 0.173 | 0.131 | 0.409 |
