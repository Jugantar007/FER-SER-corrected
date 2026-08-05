# Paired significance tests (McNemar)

Formats are evaluated on the same held-out test samples, so marginal confidence intervals are the wrong tool - they discard the pairing and are underpowered. These are paired tests on the discordant samples.

### SER — McNemar paired comparisons (n = 240)

| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp16 | 70.83 | 70.83 | +0.00 | [+0.00, +0.00] | 0 | 0 | 1.00e+00 | 1.00e+00 | ns |
| fp32 | dynrange | 70.83 | 71.25 | -0.42 | [-2.50, +1.67] | 3 | 4 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | int8_full_perchannel | 70.83 | 59.58 | +11.25 | [+6.67, +16.25] | 32 | 5 | 1.92e-05 | 3.83e-04 | *** **←** |
| fp32 | int8_full_pertensor | 70.83 | 62.50 | +8.33 | [+3.75, +12.92] | 27 | 7 | 1.12e-03 | 1.79e-02 | * |
| fp32 | qat_int8 | 70.83 | 70.00 | +0.83 | [-2.08, +3.75] | 7 | 5 | 7.74e-01 | 1.00e+00 | ns **←** |
| fp32 | ptq_finetuned_int8 | 70.83 | 65.42 | +5.42 | [+1.67, +9.17] | 17 | 4 | 7.20e-03 | 8.64e-02 | ns |
| fp16 | dynrange | 70.83 | 71.25 | -0.42 | [-2.50, +1.67] | 3 | 4 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | int8_full_perchannel | 70.83 | 59.58 | +11.25 | [+6.67, +16.25] | 32 | 5 | 1.92e-05 | 3.83e-04 | *** |
| fp16 | int8_full_pertensor | 70.83 | 62.50 | +8.33 | [+3.75, +12.92] | 27 | 7 | 1.12e-03 | 1.79e-02 | * |
| fp16 | qat_int8 | 70.83 | 70.00 | +0.83 | [-2.08, +3.75] | 7 | 5 | 7.74e-01 | 1.00e+00 | ns |
| fp16 | ptq_finetuned_int8 | 70.83 | 65.42 | +5.42 | [+1.67, +9.17] | 17 | 4 | 7.20e-03 | 8.64e-02 | ns |
| dynrange | int8_full_perchannel | 71.25 | 59.58 | +11.67 | [+7.50, +16.25] | 30 | 2 | 1.82e-06 | 3.81e-05 | *** **←** |
| dynrange | int8_full_pertensor | 71.25 | 62.50 | +8.75 | [+5.00, +12.92] | 24 | 3 | 1.19e-04 | 2.13e-03 | ** |
| dynrange | qat_int8 | 71.25 | 70.00 | +1.25 | [-1.67, +4.58] | 9 | 6 | 6.07e-01 | 1.00e+00 | ns |
| dynrange | ptq_finetuned_int8 | 71.25 | 65.42 | +5.83 | [+2.50, +9.58] | 17 | 3 | 2.58e-03 | 3.61e-02 | * |
| int8_full_perchannel | int8_full_pertensor | 59.58 | 62.50 | -2.92 | [-5.42, -0.83] | 1 | 8 | 3.91e-02 | 3.52e-01 | ns **←** |
| int8_full_perchannel | qat_int8 | 59.58 | 70.00 | -10.42 | [-15.42, -5.42] | 7 | 32 | 1.22e-04 | 2.13e-03 | ** **←** |
| int8_full_perchannel | ptq_finetuned_int8 | 59.58 | 65.42 | -5.83 | [-10.00, -1.67] | 7 | 21 | 1.40e-02 | 1.40e-01 | ns **←** |
| int8_full_pertensor | qat_int8 | 62.50 | 70.00 | -7.50 | [-12.50, -2.50] | 10 | 28 | 5.82e-03 | 7.57e-02 | ns |
| int8_full_pertensor | ptq_finetuned_int8 | 62.50 | 65.42 | -2.92 | [-7.08, +1.25] | 10 | 17 | 2.48e-01 | 1.00e+00 | ns |
| qat_int8 | ptq_finetuned_int8 | 70.00 | 65.42 | +4.58 | [+0.42, +8.75] | 20 | 9 | 6.33e-02 | 5.07e-01 | ns **←** |

`b` = A correct & B wrong; `c` = A wrong & B correct. Only discordant samples carry information.
Significance stars use Holm-adjusted p-values. **←** marks comparisons the paper's claims rest on.
