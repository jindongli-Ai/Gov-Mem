# Gemini-2.5-Flash-Lite / OpenLux corrected baseline results

91 episodes / 2218 queries per baseline. Percentages. avg.MGS is the unweighted mean of the four domain MGS values.

45 affected full episodes rerun; Mem0 reused after verification. Zero terminal execution errors and complete applicable judge labels. Internal fallback warnings are not counted as terminal execution errors and remain in source audits.

## A-Mem

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 79.52 | 51.04 | 33.33 | 25.96 |
| Office | 71.43 | 61.40 | 59.91 | 11.05 |
| Education | 35.00 | 41.67 | 61.67 | 7.83 |
| Household | 54.89 | 33.15 | 61.41 | 14.16 |
| **avg.MGS** | — | — | — | **14.75** |

## Mem0

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 42.38 | 54.69 | 61.02 | 7.49 |
| Office | 39.61 | 60.23 | 46.40 | 8.44 |
| Education | 29.44 | 40.56 | 49.44 | 8.85 |
| Household | 22.28 | 30.43 | 32.07 | 10.53 |
| **avg.MGS** | — | — | — | **8.83** |

## ReMem-I

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 63.81 | 51.04 | 37.29 | 19.59 |
| Office | 69.48 | 40.35 | 45.05 | 22.78 |
| Education | 25.00 | 29.44 | 57.78 | 7.45 |
| Household | 40.76 | 32.61 | 42.39 | 15.82 |
| **avg.MGS** | — | — | — | **16.41** |

## ReMem-S

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical | 69.05 | 42.19 | 35.59 | 25.71 |
| Office | 46.10 | 40.94 | 36.49 | 17.30 |
| Education | 10.00 | 21.11 | 43.89 | 4.43 |
| Household | 28.80 | 28.26 | 33.70 | 13.70 |
| **avg.MGS** | — | — | — | **15.28** |

Source: `outputs/gemini_baseline_network_repair_20260923`
