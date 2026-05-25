# Viterbi vs. Baseline Q-Score 对比报告

- 数据集：`/workspace/huada/moffett_data/ccf_eval`（HG002）
- 模型：`lstm_ctc_crf_finetune_moffett_fast`（0318 finetune，bf16 + quant）
- baseline：`results/0318_moffett_finetune_baseline.sam`（beam_search 解码，koi/bonito）
- viterbi ：`results/0318_moffett_finetune_viterbi.sam`（新版 viterbi 解码）
- 生成方式：`python compare_qscores.py --sam baseline.sam viterbi.sam --label baseline viterbi --output_png 0318_viterbi_vs_baseline.png --max_q 50 --min_count 200`
- 原始输出：`results/0318_viterbi_vs_baseline.log`
- 叠加图：`results/0318_viterbi_vs_baseline.png`

---

## 1. 总览

| 指标                | baseline      | viterbi       | 差异（viterbi − baseline） |
| ------------------- | ------------- | ------------- | --------------------------- |
| total reads         | 8,109         | 8,111         | +2                          |
| mapped reads        | 7,733         | 7,734         | +1                          |
| total bases (w/Q)   | 47,353,303    | 47,398,530    | +45,227 (+0.10%)            |
| Q min               | 1             | 1             | 0                           |
| Q max               | 90            | 50            | **viterbi 封顶到 50**        |
| Q mean              | 26.66         | 26.65         | −0.01                       |
| Q median            | 29.0          | 29.0          | 0                           |
| Q 25 / 75 分位       | 24.0 / 31.0   | 24.0 / 31.0   | 一致                        |
| % bases Q ≤ 1       | 0.03 %        | 0.06 %        | +0.03 pp                    |
| % bases Q ≥ 50      | 0.03 %        | 0.00 %        | −0.03 pp                    |
| **global reported Q**   | 26.85     | 26.86         | +0.01                       |
| **global empirical Q**  | **20.64** | **20.54**     | **−0.10 dB**                |
| **global 错误率**        | **0.862 %** | **0.884 %** | **+0.022 pp**               |

要点：

- Reads 数、可比对 reads 数、总 base 数、Q 的基础统计（均值/中位数/四分位）几乎一致，说明两条 pipeline 处理的是同一批 read、且 Q 的粗略分布相同。
- **viterbi 的 Q 上限被压到 50**，baseline 可达 90；对应的 Q≥50 占比 baseline 是 viterbi 的约 10 倍。模型对极高置信区的细分能力在 viterbi 路径上丢失了。
- 在 carrier/低置信端，viterbi 产生的 Q=1 占比略高一些（0.06 % vs 0.03 %），绝对量也不大。
- 全局 empirical Q：**baseline 20.64 dB，viterbi 20.54 dB**，差 0.10 dB。对应错误率分别为 0.862 % 与 0.884 %（viterbi 相对增加 ~2.6 %）。
- 从 per-read 相关性（图 (c) 的 Pearson r）看 baseline 0.560、viterbi 0.543，baseline 读级相关性也略好。

---

## 2. Per-base Q 分布（图 (a)）

```text
Reported Q   |  baseline N_tot        viterbi N_tot
-------------|-----------------------------------------
   1         |      10,084               8,644      -14.3%
   6         |     225,889             235,816      +4.4%
  10         |     342,649             337,134      -1.6%
  20         |     790,182             781,477      -1.1%
  30         |   5,028,874           5,044,191       +0.3%
  32         |   4,161,115           4,165,456       +0.1%
  40         |      29,776              23,862      -19.9%
  45         |       2,947               1,061      -64.0%
  50         |         569                  51      -91.0%
```

- 两条曲线在 Q = 6 ~ 32 这段主体区近乎重合，说明 viterbi 没有对主力分数段造成偏移。
- 越往高 Q 端 viterbi 的 bin 越少：Q ≥ 40 的总体 base 数只有 baseline 的 ~60 %，Q ≥ 45 直接降到 ~35 %，Q = 50 再降至 ~9 %。这与 viterbi 的 Q 天花板被截到 50 相符——原本被 baseline 打到 51~90 的那部分 base，viterbi 直接"沉"到了 45–50 左右，并且由于截断，高分尾部不再能表达信心差异。

---

## 3. Calibration（图 (b)，关键）

下表挑出几个代表性 Q 档（完整数据见 `0318_viterbi_vs_baseline.log`）：

| Reported Q | baseline Qemp | baseline Δ | viterbi Qemp | viterbi Δ | 谁更准 |
| ----: | ------: | ------: | ------: | ------: | :-----: |
|   2  |   7.05  | **+5.05** |  7.55  | **+5.55** | viterbi |
|   5  |   9.07  | +4.07 |  10.39  | +5.39 | viterbi |
|  10  |  12.52  | +2.52 |  14.41  | +4.41 | viterbi |
|  15  |  16.38  | +1.38 |  16.19  | +1.19 | ~ 打平 |
|  18  |  18.17  | +0.17 |  17.22  | −0.78 | baseline |
|  20  |  19.27  | −0.73 |  17.84  | −2.16 | baseline |
|  25  |  22.44  | −2.56 |  20.59  | −4.41 | baseline |
|  28  |  25.15  | −2.85 |  23.20  | −4.80 | baseline |
|  30  |  27.08  | −2.92 |  25.24  | −4.76 | baseline |
|  32  |  28.29  | −3.71 |  26.38  | −5.62 | baseline |
|  35  |  26.85  | −8.15 |  25.37  | −9.63 | baseline |
|  40  |  20.76  | −19.24 |  21.99  | −18.01 | viterbi |
|  45  |  20.89  | −24.11 |  21.23  | −23.77 | viterbi |

（Δ = 经验 Q − 报告 Q，正数表示"模型报的 Q 偏保守（低估自信）"，负数表示"偏乐观（过度自信）"。）

两种解码表现出相当不同的校准模式：

1. **低 Q 段（Q ≤ 15）：viterbi 更保守**。
   - 例如 Q = 5 时，viterbi 的真实错误率 0.0914 对应 empirical Q = 10.39，而 baseline 给出同样的 Q=5 时实际错误率 0.124 → empirical 9.07。viterbi 把"自信较低"的 base 真实品质又高估了半档左右。
   - 这反映到图 (b) 中就是蓝点在 Q=1~14 整体高于红点。

2. **中段（Q ≈ 15 ~ 18）两者交叉**，差距 < 1 dB。

3. **中高 Q 段（Q = 18 ~ 33，占据绝大多数 base）：baseline 明显更准、viterbi 更乐观**。
   - Q = 30 附近是主要峰值（~5 M base）。baseline Δ ≈ −2.9，viterbi Δ ≈ −4.8，差距接近 2 dB。
   - 也就是在模型"认为自己最稳"的区间，viterbi 的真实错误率约为 baseline 的 1.5~1.7 倍（见下表）：

| Reported Q | baseline P_err | viterbi P_err | 比例 |
| ----: | -----: | -----: | ----: |
| 25 | 0.0057 | 0.0087 | 1.53× |
| 27 | 0.0039 | 0.0060 | 1.54× |
| 28 | 0.0031 | 0.0048 | 1.55× |
| 30 | 0.0020 | 0.0030 | 1.50× |
| 32 | 0.0015 | 0.0023 | 1.53× |

4. **尾部（Q ≥ 40）两者都很差**（Δ 约 −20 ~ −31），但那里 base 数已经很少（bin 大小 10²~10⁴），统计噪声较大；viterbi 尾部 Δ 反而略优，主要是因为它根本没有产生多少 Q≥45 的 base。

---

## 4. 读级一致性（图 (c)）

- 把每条 read 的平均 Q 与它的比对恒等度（转成 identity Q = −10 log10(1 − identity)）做散点：
  - baseline：Pearson r = **0.560**，n = 7,733
  - viterbi ：Pearson r = **0.543**，n = 7,734
- 两个散点云几乎重合；baseline 的相关性略高（+0.017），与前面的校准结论一致：baseline 的 per-read 平均 Q 对 read 真实准确率更具指示性。

---

## 5. 结论 & 建议

1. **吞吐 / 分布等粗粒度指标两者无显著差异**，viterbi 可作为 baseline 的直接替换。
2. **整体准确度略逊**：viterbi 的 empirical Q 比 baseline 低 0.10 dB（错误率相对 +2.6 %）。差距不大但可重复。
3. **校准曲线形状差异明显**：
   - 低 Q 段 viterbi 偏保守；
   - 主力段 (Q ≈ 20–33) viterbi 偏乐观，错误率约为 baseline 的 1.5×；
   - viterbi 的 Q 天花板被截到 50，导致高置信尾部信息被压缩。
4. 下一步可考虑：
   - 重新检查 viterbi 分支里的 Q 计算（特别是后验概率累加与 clamp 到 [1, 50] 的逻辑）——上限为什么被压到 50，低 Q 段的 shift 是否来自平滑/先验；
   - 在 basecaller 的训练 / 校准集上拟合一次单调映射（Platt / isotonic），把 viterbi 的 Q 对齐到 baseline 的刻度；或者直接对 viterbi 复用 baseline 路径里的 Q 生成器，仅替换路径解码部分；
   - 继续对比同一批 reads 的 `*.png`（accuracy）与 `*_qval.png`（单侧校准），确认结论不是 minimap2 比对差异造成的。
