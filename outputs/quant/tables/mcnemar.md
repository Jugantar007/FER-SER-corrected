# Paired significance tests (McNemar)

Formats are evaluated on the same held-out test samples, so marginal confidence intervals are the wrong tool - they discard the pairing and are underpowered. These are paired tests on the discordant samples.

### FER — McNemar paired comparisons (n = 1948)

| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp16 | 82.49 | 82.55 | -0.05 | [-0.15, +0.00] | 0 | 1 | 1.00e+00 | 1.00e+00 | ns |
| fp32 | dynrange | 82.49 | 82.44 | +0.05 | [-0.87, +1.03] | 44 | 43 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | int8_full_perchannel | 82.49 | 61.24 | +21.25 | [+18.84, +23.77] | 544 | 130 | 5.56e-57 | 2.22e-56 | *** **←** |
| fp32 | int8_full_pertensor | 82.49 | 33.21 | +49.28 | [+46.51, +52.00] | 1083 | 123 | 7.35e-168 | 6.62e-167 | *** |
| fp16 | dynrange | 82.55 | 82.44 | +0.10 | [-0.82, +1.08] | 44 | 42 | 9.14e-01 | 1.00e+00 | ns |
| fp16 | int8_full_perchannel | 82.55 | 61.24 | +21.30 | [+18.89, +23.82] | 544 | 129 | 2.49e-57 | 1.24e-56 | *** |
| fp16 | int8_full_pertensor | 82.55 | 33.21 | +49.33 | [+46.56, +52.10] | 1083 | 122 | 2.41e-168 | 2.41e-167 | *** |
| dynrange | int8_full_perchannel | 82.44 | 61.24 | +21.20 | [+18.79, +23.66] | 536 | 123 | 5.79e-58 | 3.47e-57 | *** **←** |
| dynrange | int8_full_pertensor | 82.44 | 33.21 | +49.23 | [+46.46, +52.00] | 1085 | 126 | 7.86e-167 | 6.28e-166 | *** |
| int8_full_perchannel | int8_full_pertensor | 61.24 | 33.21 | +28.03 | [+24.85, +31.21] | 847 | 301 | 3.24e-58 | 2.27e-57 | *** **←** |

`b` = A correct & B wrong; `c` = A wrong & B correct. Only discordant samples carry information.
Significance stars use Holm-adjusted p-values. **←** marks comparisons the paper's claims rest on.

### SER — McNemar paired comparisons (n = 240)

| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp16 | 60.83 | 60.83 | +0.00 | [+0.00, +0.00] | 0 | 0 | 1.00e+00 | 1.00e+00 | ns |
| fp32 | dynrange | 60.83 | 60.42 | +0.42 | [-1.67, +2.50] | 4 | 3 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | int8_full_perchannel | 60.83 | 50.83 | +10.00 | [+5.42, +15.00] | 30 | 6 | 1.26e-04 | 1.26e-03 | ** **←** |
| fp32 | int8_full_pertensor | 60.83 | 52.08 | +8.75 | [+4.17, +13.75] | 29 | 8 | 1.01e-03 | 7.06e-03 | ** |
| fp16 | dynrange | 60.83 | 60.42 | +0.42 | [-1.67, +2.50] | 4 | 3 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | int8_full_perchannel | 60.83 | 50.83 | +10.00 | [+5.42, +15.00] | 30 | 6 | 1.26e-04 | 1.26e-03 | ** |
| fp16 | int8_full_pertensor | 60.83 | 52.08 | +8.75 | [+4.17, +13.75] | 29 | 8 | 1.01e-03 | 7.06e-03 | ** |
| dynrange | int8_full_perchannel | 60.42 | 50.83 | +9.58 | [+5.00, +14.17] | 29 | 6 | 2.00e-04 | 1.60e-03 | ** **←** |
| dynrange | int8_full_pertensor | 60.42 | 52.08 | +8.33 | [+3.75, +12.92] | 27 | 7 | 1.12e-03 | 7.06e-03 | ** |
| int8_full_perchannel | int8_full_pertensor | 50.83 | 52.08 | -1.25 | [-3.75, +1.25] | 3 | 6 | 5.08e-01 | 1.00e+00 | ns **←** |

`b` = A correct & B wrong; `c` = A wrong & B correct. Only discordant samples carry information.
Significance stars use Holm-adjusted p-values. **←** marks comparisons the paper's claims rest on.
