# Paired significance tests (McNemar)

Formats are evaluated on the same held-out test samples, so marginal confidence intervals are the wrong tool - they discard the pairing and are underpowered. These are paired tests on the discordant samples.

### SER — McNemar paired comparisons (n = 240)

| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp16 | 60.83 | 60.83 | +0.00 | [+0.00, +0.00] | 0 | 0 | 1.00e+00 | 1.00e+00 | ns |
| fp32 | dynrange | 60.83 | 60.42 | +0.42 | [-1.67, +2.50] | 4 | 3 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | int8_full_perchannel | 60.83 | 50.83 | +10.00 | [+5.42, +15.00] | 30 | 6 | 1.26e-04 | 2.65e-03 | ** **←** |
| fp32 | int8_full_pertensor | 60.83 | 52.08 | +8.75 | [+4.17, +13.75] | 29 | 8 | 1.01e-03 | 1.72e-02 | * |
| fp32 | qat_int8 | 60.83 | 60.42 | +0.42 | [-2.50, +3.33] | 6 | 5 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | ptq_finetuned_int8 | 60.83 | 55.42 | +5.42 | [+1.25, +9.58] | 21 | 8 | 2.59e-02 | 3.36e-01 | ns |
| fp16 | dynrange | 60.83 | 60.42 | +0.42 | [-1.67, +2.50] | 4 | 3 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | int8_full_perchannel | 60.83 | 50.83 | +10.00 | [+5.42, +15.00] | 30 | 6 | 1.26e-04 | 2.65e-03 | ** |
| fp16 | int8_full_pertensor | 60.83 | 52.08 | +8.75 | [+4.17, +13.75] | 29 | 8 | 1.01e-03 | 1.72e-02 | * |
| fp16 | qat_int8 | 60.83 | 60.42 | +0.42 | [-2.50, +3.33] | 6 | 5 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | ptq_finetuned_int8 | 60.83 | 55.42 | +5.42 | [+1.25, +9.58] | 21 | 8 | 2.59e-02 | 3.36e-01 | ns |
| dynrange | int8_full_perchannel | 60.42 | 50.83 | +9.58 | [+5.00, +14.17] | 29 | 6 | 2.00e-04 | 3.81e-03 | ** **←** |
| dynrange | int8_full_pertensor | 60.42 | 52.08 | +8.33 | [+3.75, +12.92] | 27 | 7 | 1.12e-03 | 1.72e-02 | * |
| dynrange | qat_int8 | 60.42 | 60.42 | +0.00 | [-2.92, +2.92] | 7 | 7 | 1.00e+00 | 1.00e+00 | ns |
| dynrange | ptq_finetuned_int8 | 60.42 | 55.42 | +5.00 | [+0.83, +9.17] | 21 | 9 | 4.46e-02 | 4.91e-01 | ns |
| int8_full_perchannel | int8_full_pertensor | 50.83 | 52.08 | -1.25 | [-3.75, +1.25] | 3 | 6 | 5.08e-01 | 1.00e+00 | ns **←** |
| int8_full_perchannel | qat_int8 | 50.83 | 60.42 | -9.58 | [-14.58, -4.58] | 10 | 33 | 7.94e-04 | 1.43e-02 | * **←** |
| int8_full_perchannel | ptq_finetuned_int8 | 50.83 | 55.42 | -4.58 | [-8.75, -0.42] | 8 | 19 | 5.43e-02 | 5.18e-01 | ns **←** |
| int8_full_pertensor | qat_int8 | 52.08 | 60.42 | -8.33 | [-13.75, -2.92] | 12 | 32 | 4.18e-03 | 5.85e-02 | ns |
| int8_full_pertensor | ptq_finetuned_int8 | 52.08 | 55.42 | -3.33 | [-7.92, +1.25] | 11 | 19 | 2.01e-01 | 1.00e+00 | ns |
| qat_int8 | ptq_finetuned_int8 | 60.42 | 55.42 | +5.00 | [+0.42, +9.58] | 22 | 10 | 5.18e-02 | 5.18e-01 | ns **←** |

`b` = A correct & B wrong; `c` = A wrong & B correct. Only discordant samples carry information.
Significance stars use Holm-adjusted p-values. **←** marks comparisons the paper's claims rest on.
