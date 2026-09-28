# OpenLux / gpt-4o-mini — corrected full results

Each method: 91 episodes / 2218 queries. All values in percent. avg.MGS is the unweighted mean of four domain MGS values.

12 full episodes rerun after recorded execution failures; earlier attempts retained. ReMem embedding/store consistency fix applied. All execution errors resolved and applicable judge labels complete.

## A-Mem

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 48.10 | 56.25 | 17.51 | 17.36 |
| Office | 32.47 | 61.99 | 24.77 | 9.28 |
| Education | 7.78 | 27.22 | 26.67 | 4.15 |
| Household | 16.85 | 17.39 | 18.48 | 11.35 |
| **avg.MGS** | — | — | — | **10.53** |

## Mem0

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 28.57 | 63.54 | 22.60 | 8.06 |
| Office | 38.31 | 66.08 | 13.96 | 11.18 |
| Education | 3.89 | 27.22 | 14.44 | 2.42 |
| Household | 7.61 | 19.57 | 10.33 | 5.49 |
| **avg.MGS** | — | — | — | **6.79** |

## ReMem-I

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 34.29 | 60.94 | 23.16 | 10.29 |
| Office | 15.58 | 43.86 | 31.08 | 6.03 |
| Education | 3.89 | 25.56 | 23.89 | 2.20 |
| Household | 7.61 | 23.37 | 16.30 | 4.88 |
| **avg.MGS** | — | — | — | **5.85** |

## ReMem-S

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 38.10 | 60.42 | 20.34 | 12.01 |
| Office | 16.23 | 50.29 | 31.53 | 5.53 |
| Education | 2.78 | 30.56 | 26.11 | 1.43 |
| Household | 8.70 | 19.57 | 14.67 | 5.97 |
| **avg.MGS** | — | — | — | **6.23** |

Source: `outputs/baselines_gpt4omini_keyrot_retry_20260923/official_metrics.json`
