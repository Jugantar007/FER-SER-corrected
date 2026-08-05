# Paired significance tests (McNemar)

Formats are evaluated on the same held-out test samples, so marginal confidence intervals are the wrong tool - they discard the pairing and are underpowered. These are paired tests on the discordant samples.

### SER — McNemar paired comparisons (n = 240)

| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp16 | 70.83 | 70.83 | +0.00 | [+0.00, +0.00] | 0 | 0 | 1.00e+00 | 1.00e+00 | ns |
| fp32 | dynrange | 70.83 | 71.25 | -0.42 | [-2.50, +1.67] | 3 | 4 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | int8_full_perchannel | 70.83 | 60.00 | +10.83 | [+5.83, +15.83] | 32 | 6 | 5.00e-05 | 1.00e-03 | ** **←** |
| fp32 | int8_full_pertensor | 70.83 | 62.92 | +7.92 | [+3.33, +12.50] | 27 | 8 | 2.35e-03 | 3.75e-02 | * |
| fp32 | qat_int8 | 70.83 | 70.00 | +0.83 | [-2.08, +3.75] | 7 | 5 | 7.74e-01 | 1.00e+00 | ns **←** |
| fp32 | ptq_finetuned_int8 | 70.83 | 65.42 | +5.42 | [+1.67, +9.17] | 17 | 4 | 7.20e-03 | 9.36e-02 | ns |
| fp16 | dynrange | 70.83 | 71.25 | -0.42 | [-2.50, +1.67] | 3 | 4 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | int8_full_perchannel | 70.83 | 60.00 | +10.83 | [+5.83, +15.83] | 32 | 6 | 5.00e-05 | 1.00e-03 | ** |
| fp16 | int8_full_pertensor | 70.83 | 62.92 | +7.92 | [+3.33, +12.50] | 27 | 8 | 2.35e-03 | 3.75e-02 | * |
| fp16 | qat_int8 | 70.83 | 70.00 | +0.83 | [-2.08, +3.75] | 7 | 5 | 7.74e-01 | 1.00e+00 | ns |
| fp16 | ptq_finetuned_int8 | 70.83 | 65.42 | +5.42 | [+1.67, +9.17] | 17 | 4 | 7.20e-03 | 9.36e-02 | ns |
| dynrange | int8_full_perchannel | 71.25 | 60.00 | +11.25 | [+7.08, +15.83] | 29 | 2 | 3.02e-06 | 6.33e-05 | *** **←** |
| dynrange | int8_full_pertensor | 71.25 | 62.92 | +8.33 | [+4.17, +12.50] | 25 | 5 | 5.23e-04 | 8.88e-03 | ** |
| dynrange | qat_int8 | 71.25 | 70.00 | +1.25 | [-1.67, +4.58] | 9 | 6 | 6.07e-01 | 1.00e+00 | ns |
| dynrange | ptq_finetuned_int8 | 71.25 | 65.42 | +5.83 | [+2.50, +9.58] | 17 | 3 | 2.58e-03 | 3.75e-02 | * |
| int8_full_perchannel | int8_full_pertensor | 60.00 | 62.92 | -2.92 | [-5.83, -0.42] | 2 | 9 | 6.54e-02 | 5.70e-01 | ns **←** |
| int8_full_perchannel | qat_int8 | 60.00 | 70.00 | -10.00 | [-15.00, -5.00] | 8 | 32 | 2.76e-04 | 4.97e-03 | ** **←** |
| int8_full_perchannel | ptq_finetuned_int8 | 60.00 | 65.42 | -5.42 | [-9.58, -1.25] | 8 | 21 | 2.59e-02 | 2.59e-01 | ns **←** |
| int8_full_pertensor | qat_int8 | 62.92 | 70.00 | -7.08 | [-12.08, -2.08] | 11 | 28 | 1.04e-02 | 1.14e-01 | ns |
| int8_full_pertensor | ptq_finetuned_int8 | 62.92 | 65.42 | -2.50 | [-6.67, +1.67] | 10 | 16 | 3.27e-01 | 1.00e+00 | ns |
| qat_int8 | ptq_finetuned_int8 | 70.00 | 65.42 | +4.58 | [+0.42, +8.75] | 20 | 9 | 6.33e-02 | 5.70e-01 | ns **←** |

`b` = A correct & B wrong; `c` = A wrong & B correct. Only discordant samples carry information.
Significance stars use Holm-adjusted p-values. **←** marks comparisons the paper's claims rest on.
