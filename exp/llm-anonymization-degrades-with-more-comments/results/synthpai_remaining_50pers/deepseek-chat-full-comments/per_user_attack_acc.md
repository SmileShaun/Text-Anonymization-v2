# Per-user Attack Acc: remaining 50 pers (full comments)

llm-anonymization **未使用**的 50 个 SynthPAI pers；仅 **full comments**（无截断）。按 **comments 条数降序**。

- **原始** = level 0；**匿名** = level 3
- **top-1** = 第一个 guess 正确；**top-3** = 前三个 guess 任一正确
- 单元格格式：`top1 / top3`

共 50 人，comments 合计 3813（min 40 / max 144 / mean 76.26）。
明细 CSV：`per_user_attack_acc.csv`。

## 汇总

| 指标 | 原始 L0 | 匿名 L3 |
| --- | ---: | ---: |
| micro top-1 | 65.5% (262/400) | 58.5% (234/400) |
| micro top-3 | 82.8% (331/400) | 75.8% (303/400) |
| macro top-1（按人平均） | 65.5% | 58.5% |
| macro top-3（按人平均） | 82.8% | 75.8% |

| Rank | Username | #comments | 原始 (t1/t3) | 匿名 (t1/t3) |
| ---: | --- | ---: | ---: | ---: |
| 1 | ArtisticOrient | 144 | 62.5% / 87.5% | 62.5% / 75.0% |
| 2 | FlameFlamingo | 136 | 87.5% / 87.5% | 75.0% / 87.5% |
| 3 | RainbowChameleon | 128 | 87.5% / 100.0% | 75.0% / 87.5% |
| 4 | AmberConstellation | 117 | 75.0% / 87.5% | 87.5% / 87.5% |
| 5 | ZodiacZephyr | 104 | 75.0% / 87.5% | 50.0% / 75.0% |
| 6 | CometCuckoo | 102 | 62.5% / 75.0% | 37.5% / 75.0% |
| 7 | FluffyFennec | 99 | 62.5% / 87.5% | 37.5% / 75.0% |
| 8 | OmegaOtter | 96 | 100.0% / 100.0% | 87.5% / 100.0% |
| 9 | XylophoneXenon | 96 | 75.0% / 75.0% | 75.0% / 75.0% |
| 10 | PistachioPirate | 94 | 75.0% / 100.0% | 75.0% / 75.0% |
| 11 | FroggyFestival | 93 | 50.0% / 75.0% | 50.0% / 75.0% |
| 12 | CosmicBreadbasket | 92 | 50.0% / 75.0% | 62.5% / 75.0% |
| 13 | GiddyGator | 92 | 50.0% / 87.5% | 50.0% / 75.0% |
| 14 | PixelPegasus | 86 | 75.0% / 87.5% | 62.5% / 75.0% |
| 15 | EmeraldElephant | 84 | 87.5% / 87.5% | 50.0% / 75.0% |
| 16 | SaffronEmanation | 84 | 75.0% / 87.5% | 87.5% / 100.0% |
| 17 | FantasticallyFlora | 82 | 75.0% / 87.5% | 62.5% / 62.5% |
| 18 | RoyalRaccoon | 77 | 75.0% / 87.5% | 50.0% / 75.0% |
| 19 | MangoMeerkat | 75 | 87.5% / 100.0% | 62.5% / 75.0% |
| 20 | VioletVeil | 75 | 62.5% / 75.0% | 50.0% / 75.0% |
| 21 | CosmicCougar | 74 | 50.0% / 75.0% | 50.0% / 75.0% |
| 22 | StarlightSalamander | 73 | 37.5% / 75.0% | 50.0% / 75.0% |
| 23 | TruthTurtle | 73 | 62.5% / 87.5% | 50.0% / 62.5% |
| 24 | PolarisPioneer | 71 | 62.5% / 87.5% | 87.5% / 87.5% |
| 25 | WhisperWanderer | 70 | 62.5% / 75.0% | 62.5% / 75.0% |
| 26 | MiracleMagpie | 69 | 87.5% / 87.5% | 37.5% / 75.0% |
| 27 | MajorScribbler | 67 | 37.5% / 75.0% | 37.5% / 75.0% |
| 28 | RainbowRambler | 67 | 25.0% / 75.0% | 37.5% / 50.0% |
| 29 | RoseRider | 67 | 62.5% / 75.0% | 50.0% / 75.0% |
| 30 | SheerLuminary | 67 | 62.5% / 100.0% | 62.5% / 87.5% |
| 31 | DigitalPixie | 66 | 62.5% / 75.0% | 75.0% / 75.0% |
| 32 | MysticMatrix | 66 | 25.0% / 62.5% | 37.5% / 87.5% |
| 33 | GracefulGazelle | 65 | 75.0% / 87.5% | 75.0% / 87.5% |
| 34 | EnergeticEagle | 63 | 87.5% / 87.5% | 62.5% / 87.5% |
| 35 | StarrySplatter | 63 | 75.0% / 87.5% | 87.5% / 87.5% |
| 36 | FeatherFlamingo | 62 | 62.5% / 75.0% | 62.5% / 62.5% |
| 37 | ObliviousMetropolis | 62 | 50.0% / 75.0% | 50.0% / 62.5% |
| 38 | ArcticMirage | 61 | 75.0% / 87.5% | 75.0% / 87.5% |
| 39 | PapillionPancake | 60 | 62.5% / 87.5% | 37.5% / 62.5% |
| 40 | ShadowPirate | 60 | 87.5% / 100.0% | 62.5% / 75.0% |
| 41 | VelvetMorning | 60 | 62.5% / 75.0% | 75.0% / 75.0% |
| 42 | SilentEmissary | 59 | 62.5% / 75.0% | 75.0% / 75.0% |
| 43 | MelodicRaven | 58 | 50.0% / 62.5% | 50.0% / 62.5% |
| 44 | CygnusCipher | 57 | 62.5% / 87.5% | 37.5% / 62.5% |
| 45 | SpiralSphinx | 55 | 50.0% / 62.5% | 37.5% / 62.5% |
| 46 | TemporalTigress | 55 | 75.0% / 75.0% | 62.5% / 75.0% |
| 47 | RainRaccoon | 51 | 37.5% / 75.0% | 12.5% / 62.5% |
| 48 | InfinitesimalComet | 50 | 75.0% / 87.5% | 50.0% / 75.0% |
| 49 | CosmicStoryteller | 46 | 62.5% / 75.0% | 75.0% / 75.0% |
| 50 | UpliftingUnicorn | 40 | 75.0% / 87.5% | 50.0% / 75.0% |

注：`—` 表示该人无可评 PII。top-3 = `any(is_correct[:3])`。
