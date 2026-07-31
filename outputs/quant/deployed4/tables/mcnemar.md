# Paired significance tests (McNemar)

Formats are evaluated on the same held-out test samples, so marginal confidence intervals are the wrong tool - they discard the pairing and are underpowered. These are paired tests on the discordant samples.

### SER — McNemar paired comparisons (n = 240)

| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp16 | 70.83 | 70.83 | +0.00 | [+0.00, +0.00] | 0 | 0 | 1.00e+00 | 1.00e+00 | ns |
| fp32 | dynrange | 70.83 | 71.25 | -0.42 | [-2.50, +1.67] | 3 | 4 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | int8_full_perchannel | 70.83 | 59.58 | +11.25 | [+6.67, +16.25] | 32 | 5 | 1.92e-05 | 1.73e-04 | *** **←** |
| fp32 | int8_full_pertensor | 70.83 | 62.50 | +8.33 | [+3.75, +12.92] | 27 | 7 | 1.12e-03 | 6.72e-03 | ** |
| fp16 | dynrange | 70.83 | 71.25 | -0.42 | [-2.50, +1.67] | 3 | 4 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | int8_full_perchannel | 70.83 | 59.58 | +11.25 | [+6.67, +16.25] | 32 | 5 | 1.92e-05 | 1.73e-04 | *** |
| fp16 | int8_full_pertensor | 70.83 | 62.50 | +8.33 | [+3.75, +12.92] | 27 | 7 | 1.12e-03 | 6.72e-03 | ** |
| dynrange | int8_full_perchannel | 71.25 | 59.58 | +11.67 | [+7.50, +16.25] | 30 | 2 | 1.82e-06 | 1.82e-05 | *** **←** |
| dynrange | int8_full_pertensor | 71.25 | 62.50 | +8.75 | [+5.00, +12.92] | 24 | 3 | 1.19e-04 | 8.30e-04 | *** |
| int8_full_perchannel | int8_full_pertensor | 59.58 | 62.50 | -2.92 | [-5.42, -0.83] | 1 | 8 | 3.91e-02 | 1.56e-01 | ns **←** |

`b` = A correct & B wrong; `c` = A wrong & B correct. Only discordant samples carry information.
Significance stars use Holm-adjusted p-values. **←** marks comparisons the paper's claims rest on.
