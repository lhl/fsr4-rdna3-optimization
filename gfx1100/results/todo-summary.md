# gfx1100 TODO Result Summary

## INT8 Comparable Ranking

| rank | config | mean_ms | median_ms | cv% | delta% vs ref | verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `todo-p1-candA-int8-items4-trials3` | 0.006836 | 0.006826 | 1.197 | 11.526 | keep |
| 2 | `todo-p1-candC-int8-items4-inner16-trials3` | 0.006838 | 0.006824 | 0.543 | 11.552 | keep |
| 3 | `todo-p1-items-int8-4` | 0.006853 | 0.006842 | 0.500 | 11.323 | keep |
| 4 | `todo-p1-items-int8-4-trials3` | 0.006864 | 0.006849 | 1.273 | 11.237 | keep |
| 5 | `todo-p1-items-int8-2` | 0.007389 | 0.007377 | 0.384 | 4.389 | keep |
| 6 | `todo-p1-threads-int8-192-trials3` | 0.007661 | 0.007769 | 3.620 | -0.691 | unsure |
| 7 | `todo-p0-o14-int8-inner16` | 0.007709 | 0.007687 | 0.572 | 0.372 | unsure |
| 8 | `todo-p1-items-int8-1` | 0.007712 | 0.007687 | 0.579 | 0.372 | unsure |
| 9 | `todo-p0-o14-int8-inner32` | 0.007714 | 0.007694 | 0.582 | 0.281 | unsure |
| 10 | `todo-p0-o14-int8-inner16-trials3` | 0.007730 | 0.007707 | 0.581 | 0.108 | unsure |
| 11 | `todo-p1-candB-int8-inner16-trials3` | 0.007756 | 0.007732 | 0.580 | -0.207 | unsure |
| 12 | `todo-p1-threads-int8-192` | 0.007760 | 0.007738 | 0.578 | -0.289 | unsure |

## FP8 Comparable Ranking

| rank | config | mean_ms | median_ms | cv% | delta% vs ref | verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `todo-p0-o14-fp8-inner8` | 0.009710 | 0.009942 | 4.319 | 6.513 | keep |
| 2 | `todo-p1-candC-fp8-items1-inner8-trials3` | 0.009958 | 0.010001 | 1.801 | 5.962 | keep |
| 3 | `todo-p1-candB-fp8-inner8-trials3` | 0.009983 | 0.009997 | 1.452 | 5.993 | keep |
| 4 | `todo-p0-o14-fp8-inner8-trials3` | 0.010005 | 0.009996 | 0.743 | 6.002 | keep |
| 5 | `todo-p1-threads-fp8-512` | 0.010599 | 0.010588 | 0.822 | 0.439 | unsure |
| 6 | `todo-p0-o14-fp8-inner16` | 0.010605 | 0.010593 | 0.848 | 0.392 | unsure |
| 7 | `todo-p1-candA-fp8-items1-trials3` | 0.010614 | 0.010651 | 1.783 | -0.157 | unsure |
| 8 | `todo-p1-threads-fp8-512-trials3` | 0.010641 | 0.010631 | 0.828 | 0.031 | unsure |
| 9 | `todo-p1-threads-fp8-192` | 0.010663 | 0.010837 | 3.643 | -1.903 | unsure |
| 10 | `todo-p1-threads-fp8-384` | 0.010709 | 0.010699 | 0.809 | -0.605 | unsure |
| 11 | `todo-p1-threads-fp8-320` | 0.010813 | 0.010800 | 0.873 | -1.555 | unsure |
| 12 | `todo-p1-threads-fp8-1024` | 0.011092 | 0.011082 | 0.744 | -4.206 | drop |

## Mixed Policy Ranking

| rank | policy | int8_ms | fp8_ms | total_ms | total delta% vs canonical |
|---:|---|---:|---:|---:|---:|
| 1 | candidate-C (A+B combined) | 0.006838 | 0.009958 | 0.016796 | 8.485 |
| 2 | candidate-A (int8 items=4, fp8 items=1) | 0.006836 | 0.010614 | 0.017450 | 4.922 |
| 3 | candidate-B (int8 inner=16, fp8 inner=8) | 0.007756 | 0.009983 | 0.017739 | 3.347 |
| 4 | canonical-default-both | 0.007713 | 0.010641 | 0.018353 | 0.000 |

