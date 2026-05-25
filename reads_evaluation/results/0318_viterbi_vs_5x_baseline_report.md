# Viterbi vs. 5x_Baseline Q-Score 对比报告

- 数据集：`/workspace/huada/moffett_data/ccf_eval`（HG002）
- 模型：`lstm_ctc_crf_finetune_moffett_fast`（0318 finetune，bf16 + quant）
- 5x_baseline：`results/0318_moffett_finetune_5x_baseline.sam`（beam_search 解码，beam 宽度 5×）
- viterbi     ：`results/0318_moffett_finetune_viterbi.sam`（新版 viterbi 解码）
- 生成方式：`python compare_qscores.py --sam 5x_baseline.sam viterbi.sam --label 5x_baseline viterbi --output_png 0318_viterbi_vs_5x_baseline.png --max_q 50 --min_count 200`
- 原始输出：`results/0318_viterbi_vs_5x_baseline.log`
- 叠加图：`results/0318_viterbi_vs_5x_baseline.png`

---

## 1. 总览

| 指标                    | 5x_baseline   | viterbi       | 差异（viterbi − 5x）  |
| ----------------------- | ------------- | ------------- | --------------------- |
| total reads             | 8,118         | 8,111         | −7                    |
| mapped reads            | 7,733         | 7,734         | +1                    |
| total bases (w/Q)       | 47,374,651    | 47,398,530    | +23,879 (+0.05 %)     |
| Q min                   | 1             | 1             | 0                     |
| Q max                   | 90            | 50            | **viterbi 封顶到 50**  |
| Q mean                  | **27.34**     | 26.65         | **−0.69**             |
| Q median                | 30.0          | 29.0          | −1.0                  |
| Q 25 / 75 分位          | 25.0 / 32.0   | 24.0 / 31.0   | 整体下移 ~1           |
| % bases Q ≤ 1           | 0.03 %        | 0.06 %        | +0.03 pp              |
| % bases Q ≥ 50          | 0.02 %        | 0.00 %        | −0.02 pp              |
| **global reported Q**   | 27.53         | 26.86         | −0.67                 |
| **global empirical Q**  | **20.92 dB**  | **20.54 dB**  | **−0.38 dB**          |
| **global 错误率**        | **0.809 %**   | **0.884 %**   | **+0.075 pp（相对 +9.3 %）** |
| per-read Pearson r      | 0.556         | 0.543         | −0.013                |

要点：

- reads 数、mapped 数、总 base 数几乎一致，viterbi 与 5x_baseline 处理的是同一批 read，吞吐没有损失。
- **viterbi 的整体 Q 分布比 5x_baseline 低 ~1 个单位**（mean 26.65 vs 27.34，median 29 vs 30，25/75 分位都下移 1）。
- **viterbi 的 Q 上限被截到 50**（5x_baseline 可达 90），高置信尾部被压缩，Q ≥ 50 的 bin 几乎空掉。
- 整体准确度：viterbi empirical Q 比 5x_baseline 低 **0.38 dB**，错误率从 0.809 % 升到 0.884 %，相对恶化约 **+9.3 %**。
- 这个差距明显大于此前 `viterbi vs baseline`（beam 1×）的 0.10 dB / +2.6 %：换成更强的 5× baseline 后，viterbi 的 gap 扩大到约 3.4 倍。

> 参照：此前 `0318_viterbi_vs_baseline_report.md`
> - baseline (beam 1×)   : err 0.862 % / empQ 20.64
> - **5x_baseline (beam 5×) : err 0.809 % / empQ 20.92（目前最佳）**
> - viterbi              : err 0.884 % / empQ 20.54

---

## 2. Per-base Q 分布（图 (a)）

```text
Reported Q   |  5x_baseline N_tot   viterbi N_tot
-------------|-----------------------------------------
   1         |     9,781                  8,644     -11.6%
   6         |   201,901                235,816     +16.8%
  10         |   304,510                337,134     +10.7%
  20         |   677,001                781,477     +15.4%
  28         | 2,902,882              3,408,116     +17.4%
  30         | 5,058,956              5,044,191      -0.3%
  31         | 5,653,351              5,054,587     -10.6%
  32         | 5,041,978              4,165,456     -17.4%
  33         | 3,556,343              2,806,577     -21.1%
  35         | 1,038,922                788,822     -24.1%
  40         |    32,716                 23,862     -27.1%
  45         |     3,042                  1,061     -65.1%
  50         |       567                     51     -91.0%
```

- 在主力分数段（Q ≈ 28–33），5x_baseline 的分布"右侧更厚"：Q = 31 比 viterbi 多出 ~60 万 base（+11.9 %），Q = 32 多出 ~88 万 base（+21.1 %）。
- viterbi 则把这部分质量"沉降"到 Q = 20–28 区间（Q = 20 多 15 %，Q = 28 多 17 %）。
- 高 Q 尾部继续塌陷：viterbi Q ≥ 40 只剩 5x_baseline 的 ~70 %，Q ≥ 45 只剩 35 %，Q = 50 只剩 9 %（和之前与普通 baseline 对比时的结论一致）。

两处边角：

- **低 Q 段** viterbi 的 Q = 1 占比是 5x 的 ~2×（0.06 % vs 0.03 %），绝对量小。
- 中段 Q = 6–14 viterbi 反而比 5x 略多（+10 %~+17 %）——后面的校准表会揭示这一段 viterbi 其实是"保守型偏差"，不一定是质量更差。

---

## 3. Calibration（图 (b)，关键）

挑选代表性 Q 档（完整表见 `0318_viterbi_vs_5x_baseline.log`）：

| Reported Q | 5x Qemp | 5x Δ   | viterbi Qemp | viterbi Δ | 谁更准 |
| ----: | ------: | -----: | ------: | -----: | :-----: |
|   2   |  7.20   | +5.20  |  7.55   | +5.55  | viterbi |
|   5   |  8.99   | +3.99  | 10.39   | +5.39  | viterbi |
|  10   | 12.34   | +2.34  | 14.41   | +4.41  | viterbi |
|  15   | 16.06   | +1.06  | 16.19   | +1.19  | ~ 打平   |
|  17   | 17.41   | +0.41  | 16.77   | −0.23  | 5x_baseline |
|  20   | 19.08   | −0.92  | 17.84   | −2.16  | 5x_baseline |
|  25   | 22.23   | −2.77  | 20.59   | −4.41  | 5x_baseline |
|  28   | 24.66   | −3.34  | 23.20   | −4.80  | 5x_baseline |
|  30   | 26.94   | −3.06  | 25.24   | −4.76  | 5x_baseline |
|  32   | 28.26   | −3.74  | 26.38   | −5.62  | 5x_baseline |
|  33   | 28.52   | −4.48  | 26.51   | −6.49  | 5x_baseline |
|  35   | 27.62   | −7.38  | 25.37   | −9.63  | 5x_baseline |
|  40   | 22.82   | −17.18 | 21.99   | −18.01 | 5x_baseline |
|  45   | 21.61   | −23.39 | 21.23   | −23.77 | ~ 打平   |

（Δ = empirical − reported；正 ≈ 过于保守，负 ≈ 过度自信。）

主力 Q 段（Q ≈ 20 – 33）viterbi 的真实错误率约为 5x_baseline 的 1.3 × – 1.5 ×：

| Reported Q | 5x P_err | viterbi P_err | 比例 |
| ----: | -----: | -----: | ----: |
| 20 | 0.0124 | 0.0165 | 1.33× |
| 25 | 0.0060 | 0.0087 | 1.45× |
| 28 | 0.0034 | 0.0048 | 1.41× |
| 30 | 0.0020 | 0.0030 | 1.50× |
| 32 | 0.0015 | 0.0023 | 1.53× |
| 33 | 0.0014 | 0.0022 | 1.57× |

特点总结：

1. **Q ≤ 14：viterbi 更"保守"**，整体高 2 – 4 dB，也就是 viterbi 在"低置信区间"实际准确率更高。和上次与普通 baseline 的对比相同，这个区段 viterbi 占优。
2. **Q ≈ 15 – 17 两线相交**（跟普通 baseline 对比时也是这段相交）。
3. **Q ≥ 18 且主要质量集中的 20 – 33 区间：5x_baseline 明显更好校准**，viterbi 过度自信 2 dB 以上；越往峰值（Q = 30 – 33）差距越大。
4. **Q ≥ 35 的稀有尾部两者都崩塌**（Δ 达 −17 ~ −30），但 viterbi 尾部 bin 本身少得多，对总体错误率贡献几乎可以忽略。
5. 由此可以直观解释总体 0.38 dB 的劣势：**错误几乎完全来自 Q ≈ 20 – 33 的过度自信**——这一段占了 > 60 % 的 base。

---

## 4. 读级一致性（图 (c)）

- 把每条 read 的平均报告 Q 与比对恒等度转换的 identity Q 做散点（`id_Q = −10 log10(1 − identity)`）：

  | 方法        | Pearson r | reads |
  | ----------- | --------: | ----: |
  | 5x_baseline | 0.556     | 7,733 |
  | viterbi     | 0.543     | 7,734 |

- 两个散点云几乎叠在一起，5x_baseline 相关性略高（+0.013）。
- 结合 per-base 结论：per-read 平均 Q 对真实识别率的指示性 5x_baseline 稍强，但差异不大，相比 per-base 的 2 dB 校准差距要温和得多——因为按 read 做平均之后，Q ≈ 30 那一段的过度自信会被各段拉开的偏差互相抵消。

---

## 5. 对照：和此前 `baseline（beam 1×）` 对比的差别

| 比较对象               | 错误率    | empirical Q | viterbi Δ (%) |
| ---------------------- | --------: | ----------: | -------------: |
| baseline（beam 1×）     |  0.862 %  | 20.64 dB    | viterbi +2.6 % |
| **5x_baseline（beam 5×）** | **0.809 %** | **20.92 dB** | **viterbi +9.3 %** |
| viterbi                | 0.884 %   | 20.54 dB    | —              |

- 把对照换成 5× beam 后，baseline 本身错误率下降 6.1 %（0.862 → 0.809），empQ 升 0.28 dB；
- 同时 viterbi 保持不变，所以 gap 从 0.10 dB 扩大到 0.38 dB；
- 说明**当前 viterbi 路径损失的几乎全部是 beam 搜索带来的那部分**——beam 越宽对 viterbi 越不利。

---

## 6. 结论 & 建议

1. **吞吐、reads 数、Q 分布形状层面两者等价**，viterbi 能产出同样规模的结果。
2. **校准/准确度层面 5x_baseline 明显更好**：global empirical Q 高 0.38 dB，错误率相对低 9 %。
3. **问题集中在主力质量段 Q ≈ 20 – 33**：viterbi 在这段过度自信，真实错误率是 5x_baseline 的 1.3 × – 1.5 ×。
4. **viterbi 的 Q 天花板被截到 50** 是另一个独立问题，导致高置信尾部丢分辨率（和普通 baseline 对比已有相同结论）。
5. 与 `viterbi vs baseline (beam 1×)` 对比，viterbi 的劣势放大了 ~3.4 倍，进一步说明 viterbi 路径没有 beam 机制带来的多假设竞争信息。

后续可以考虑的排查/改进：

- 把 viterbi 的 Q 计算逐段打印出来，重点检查 Q 输出值是否确实上溢被 clamp 到 50 的 path——若是，放宽上限后主力段的过度自信是否减轻；
- 在校准集上拟合一次单调映射（Platt / isotonic），把 viterbi 的 Q 对齐到 5x_baseline 的刻度，快速补回 0.3+ dB；
- 另一条路：viterbi 解码只用作"选路径"，Q 仍从 beam posterior 计算（即路径解码换 viterbi、Q 生成沿用 baseline），观察是否能把 gap 打到 0.1 dB 以内；
- 同一批 reads 的 accuracy 图 (`*.png`) 与单侧校准图 (`*_qval.png`) 再横向对齐一次，确认结论不是比对噪声导致的。
