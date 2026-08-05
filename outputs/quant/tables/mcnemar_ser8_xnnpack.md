# Paired significance tests (McNemar)

Formats are evaluated on the same held-out test samples, so marginal confidence intervals are the wrong tool - they discard the pairing and are underpowered. These are paired tests on the discordant samples.

### SER — McNemar paired comparisons (n = 240)

| A | B | acc A | acc B | Δ pts | paired 95% CI | b | c | p | p (Holm) | |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | fp16 | 60.83 | 60.83 | +0.00 | [+0.00, +0.00] | 0 | 0 | 1.00e+00 | 1.00e+00 | ns |
| fp32 | dynrange | 60.83 | 60.42 | +0.42 | [-1.67, +2.50] | 4 | 3 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | int8_full_perchannel | 60.83 | 50.42 | +10.42 | [+5.83, +15.42] | 32 | 7 | 1.22e-04 | 2.55e-03 | ** **←** |
| fp32 | int8_full_pertensor | 60.83 | 51.25 | +9.58 | [+4.58, +14.58] | 31 | 8 | 4.27e-04 | 7.69e-03 | ** |
| fp32 | qat_int8 | 60.83 | 60.42 | +0.42 | [-2.50, +3.33] | 6 | 5 | 1.00e+00 | 1.00e+00 | ns **←** |
| fp32 | ptq_finetuned_int8 | 60.83 | 55.00 | +5.83 | [+1.25, +10.42] | 22 | 8 | 1.76e-02 | 2.29e-01 | ns |
| fp16 | dynrange | 60.83 | 60.42 | +0.42 | [-1.67, +2.50] | 4 | 3 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | int8_full_perchannel | 60.83 | 50.42 | +10.42 | [+5.83, +15.42] | 32 | 7 | 1.22e-04 | 2.55e-03 | ** |
| fp16 | int8_full_pertensor | 60.83 | 51.25 | +9.58 | [+4.58, +14.58] | 31 | 8 | 4.27e-04 | 7.69e-03 | ** |
| fp16 | qat_int8 | 60.83 | 60.42 | +0.42 | [-2.50, +3.33] | 6 | 5 | 1.00e+00 | 1.00e+00 | ns |
| fp16 | ptq_finetuned_int8 | 60.83 | 55.00 | +5.83 | [+1.25, +10.42] | 22 | 8 | 1.76e-02 | 2.29e-01 | ns |
| dynrange | int8_full_perchannel | 60.42 | 50.42 | +10.00 | [+5.42, +14.58] | 30 | 6 | 1.26e-04 | 2.55e-03 | ** **←** |
| dynrange | int8_full_pertensor | 60.42 | 51.25 | +9.17 | [+4.58, +13.75] | 29 | 7 | 4.65e-04 | 7.69e-03 | ** |
| dynrange | qat_int8 | 60.42 | 60.42 | +0.00 | [-2.92, +2.92] | 7 | 7 | 1.00e+00 | 1.00e+00 | ns |
| dynrange | ptq_finetuned_int8 | 60.42 | 55.00 | +5.42 | [+0.83, +10.00] | 22 | 9 | 3.11e-02 | 3.43e-01 | ns |
| int8_full_perchannel | int8_full_pertensor | 50.42 | 51.25 | -0.83 | [-3.33, +1.67] | 4 | 6 | 7.54e-01 | 1.00e+00 | ns **←** |
| int8_full_perchannel | qat_int8 | 50.42 | 60.42 | -10.00 | [-15.42, -4.58] | 11 | 35 | 6.96e-04 | 1.04e-02 | * **←** |
| int8_full_perchannel | ptq_finetuned_int8 | 50.42 | 55.00 | -4.58 | [-9.17, +0.00] | 10 | 21 | 7.25e-02 | 6.52e-01 | ns **←** |
| int8_full_pertensor | qat_int8 | 51.25 | 60.42 | -9.17 | [-14.58, -3.75] | 12 | 34 | 1.96e-03 | 2.74e-02 | * |
| int8_full_pertensor | ptq_finetuned_int8 | 51.25 | 55.00 | -3.75 | [-8.33, +0.83] | 11 | 20 | 1.51e-01 | 1.00e+00 | ns |
| qat_int8 | ptq_finetuned_int8 | 60.42 | 55.00 | +5.42 | [+0.83, +10.00] | 23 | 10 | 3.67e-02 | 3.67e-01 | ns **←** |

`b` = A correct & B wrong; `c` = A wrong & B correct. Only discordant samples carry information.
Significance stars use Holm-adjusted p-values. **←** marks comparisons the paper's claims rest on.
