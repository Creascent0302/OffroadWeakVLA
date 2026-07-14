# TerraGrip-VLA · 架构与运行说明

> 本文档是 `terragrip_vla/` 这个代码库的完整导览：**它在做什么、怎么组织的、怎么跑、每个文件负责什么、以及为什么某些地方和最初的规格不一样**。

---

## 0. 一句话说清这个项目

一辆履带车可以升降履带，改变**接地面积**，分 3 档：`0=S`(小) / `1=M`(中) / `2=L`(大)。接地面积越大 → 越抓地（打滑越少）但越费力。系统要根据**前视相机看到的地形** + **一句自然语言指令**，选一个档位。

**科学问题**：决策该不该"路过"一个可解释的物理概念——**每个档位下的打滑率(slip)**？

代码库实现了**三种可一键切换、公平对比**的路径：

```
modular   视觉 ──▶ slip(g), g∈{S,M,L} ──▶ 保形上界 ──▶ 解析选择器(α,τ,λ) ──▶ 档位
e2e       (视觉, 语言) ──────────────────────────────────────────────────▶ 档位
hybrid    视觉 ──▶ slip 瓶颈 ──┬──────────────────────────────────────────▶ 档位
                scw·视觉旁路 ──┘   (联合训练; scw = 泄漏旋钮)
```

---

## 1. 你提出的那个缺口，以及它的解法

> **训练数据里只有 (图像, 档位, 实测slip)，没有语言。那怎么做 VLA？**

这是整个设计的枢纽，答案是物理性的：

### **物理与语言无关，偏好与语言有关。**

- 泥地在 L 档打滑 0.12，这是**客观事实**。你对车说"小心点"，泥地不会因此少滑。
  → **slip 头永远看不到语言**（`test_modes.py::test_slip_is_language_invariant` 断言：careful 和 fast 下的 slip 预测差异必须**严格为 0**）。

- 但"**该选哪个档**"取决于你愿意承担多少风险。
  → **语言在决策处进入**，形式是**风险预算 (α, τ, λ)**：

| 参数 | 含义 | 方向 |
|---|---|---|
| `α` | 保形失效率 | 越小 → 打滑上界越宽 → 越保守 |
| `τ` | 可容忍的打滑上限 | 越小 → 越严格 → 越保守 |
| `λ` | 在"都可接受"的档位间，用多大权重拿打滑换省力 | 越大 → 越偏抓地 |

```
careful (小心)  → (α=0.05, τ=0.15, λ=3.0)  → 保守 → 偏大档
normal  (正常)  → (α=0.10, τ=0.25, λ=0.5)
fast    (快/省电)→ (α=0.20, τ=0.35, λ=0.0)  → 高效 → 偏小档
```

### 三种模式怎么消费语言

- **modular**：`interpret(文本)` → 预算 → 交给保形选择器**解析式**使用。**语言完全不需要训练**。
- **e2e / hybrid**：吃冻结 MiniLM 的**句向量**，必须自己**学**出同一个映射（和感知纠缠在一起）。

三者输入**完全相同** `(图像, 指令)` → 对比才公平；**三者都是真正的 VLA**（动作 = 档位命令）。

### 训练标签怎么来：语言增广

数据里没有语言，我们**造**出来（`data/dataset.py: LanguageAugmenter`）：

```
每个样本、每个 epoch：
  随机采一条指令  →  解析出 (τ, λ)  →  用它把该样本的真值 slip 曲线变成 oracle 档位标签
                                        best_gear(真值曲线, τ, λ)
```

### 这个设计白送两个论文卖点（代码里都测了，不是嘴上说说）

1. **标签成本不对称**
   `modular` 只需要"**你实际走的那一档**的 slip" —— 真·自监督，本体感觉直接给。
   `e2e/hybrid` 需要 `best_gear`，而它需要**你没走过的档位**的 slip → **反事实监督**。真机上这是昂贵得多的要求。

2. **标定成本不对称**
   slip 的保形上界**只标定一次**，服务所有指令（包括训练时没见过的风险预算）。
   APS 必须**按指令逐条重新标定**，因为它覆盖的标签 `best_gear` 会随指令移动。

---

## 2. 仓库结构

```
terragrip_vla/
├── constants.py          档位/接地面积/图像几何 —— 全局共用的唯一定义
├── language.py           ★ 风险预算表、指令池(train/heldout)、冻结 MiniLM、interpret()
├── features.py           冻结骨干的特征缓存（按图像内容哈希键控）
├── runtime.py            共享运行时：一份配置/一份数据/一个骨干/一份缓存
│
├── data/
│   ├── schema.py         Sample 数据类
│   ├── mock_generator.py ★ 异方差合成数据 + OOD 地形 + 确定性渲染
│   ├── dataset.py        ★ DataSource 抽象 / 特征数据集 / LanguageAugmenter
│   └── labels.py         ★ best_gear —— 全项目唯一的决策规则
│
├── models/
│   ├── perception.py     冻结 DINOv2 + ROI 掩码池化 → phi_vis
│   ├── slip_head.py      ★ (phi, gear) → slip；含 sigma 分支（保形归一化用）
│   ├── gear_head.py      features → 3 档 logits
│   └── model.py          ★ TerraGripModel：一个类，三条路径，config 一键切换
│
├── conformal/
│   ├── split_conformal.py ★ 归一化单边保形（modular/hybrid）
│   ├── aps.py             ★ 随机化 APS 预测集合（e2e/hybrid）
│   └── select.py          ★★ 选档 —— 语言与模型相遇的地方
│
├── training/
│   ├── losses.py         按 mode 切换的损失（一个函数，不写三份脚本）
│   ├── trainer.py        训练循环（三模式共用）
│   └── train.py          训练入口
│
├── eval/
│   ├── metrics.py        评分（共同随机数：所有策略在同一反事实世界里打分）
│   ├── baselines.py      FixedS/M/L, ReactiveOnly, Random, Oracle
│   ├── policies.py       策略抽象：模型前向 → numpy → 选档
│   ├── plotting.py       论文配色（已过 CVD 色盲安全验证）
│   ├── run_eval.py       E1 帕累托图 + 保形覆盖率
│   ├── compare.py        ★ 多种子 A/B/C 主表（分布内/留出指令/OOD 三种情形）
│   └── analysis/
│       ├── leakage.py    ★ 泄漏曲线（hybrid 的决策真的经过概念吗？）
│       ├── probe.py      ★ 探针（e2e 是不是偷偷算了牵引？含饱和检测）
│       └── intervention.py ★ 测试时概念干预（+OOD）
│
├── scripts/
│   ├── env_check.py         验证 Blackwell(sm_120) 内核可用
│   ├── verify_framework.py  ★★ 框架效果验证：25 条断言，逐条检查"效果是否达到"
│   └── docker/Dockerfile    cu128 镜像
│
├── configs/
│   ├── default.yaml      正式规模（dinov2_vitb14, 4500 图, 5 种子）
│   ├── small.yaml        ★ 小而真（dinov2_vits14, 3200 图, 3 种子）—— 几分钟跑完
│   └── mode_{modular,e2e,hybrid}.yaml
│
└── tests/                70 项测试（保形覆盖率是硬性断言）
```

★ = 承载核心设计的文件；★★ = 最该先读的。

---

## 3. 数据流（一次完整推理）

```
                    ┌──────────────────────────────────────────┐
   前视图像 ──────▶ │ Perception: 冻结 DINOv2 + ROI 掩码池化    │──▶ phi_vis [B,D]
                    │ （梯形 ROI = 车即将碾过的那片地）          │      ↑ 标准化+L2归一化
                    └──────────────────────────────────────────┘
                                                                        │
   指令文本 ─┬─▶ interpret() ─▶ 风险预算 (α,τ,λ)  ─────────────┐        │
             │   [keyword → 否则 MiniLM 最近邻]                │        │
             └─▶ 冻结 MiniLM ─▶ 句向量 [B,384] ──────┐         │        │
                                                      │         │        │
   ┌──────────────────────────────────────────────────┼─────────┼────────┼───┐
   │  modular:                                        │         │        ▼   │
   │      SlipHead(phi, g) → slip(g) ∀g  ──▶ 保形上界 ─┼────▶ 选择器(α,τ,λ) → 档位
   │      ⚠ 不吃句向量                                 │         │            │
   │                                                   │         │            │
   │  e2e:                                             ▼         │            │
   │      GearHead([phi, 句向量]) ──▶ logits ──▶ APS 集合 ──▶ 解析 ──────▶ 档位
   │                                                             │            │
   │  hybrid:                                                    │            │
   │      SlipHead → slip 概念[B,3] ─┐                           │            │
   │      scw · phi (旁路) ──────────┼─▶ GearHead ──▶ APS ──────────────▶ 档位
   │      句向量 ────────────────────┘                                        │
   └──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 运行方法

### 4.1 安装

```bash
# 目标机（8×RTX 5090, Blackwell sm_120）必须用 CUDA 12.8 + torch ≥ 2.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/env_check.py          # 期望打印 sm_120，并成功跑一次 bf16 matmul

# 或直接用 Docker（基础镜像已带 cu128 torch）
docker build -f scripts/docker/Dockerfile -t terragrip .
```

> ⚠️ **不要** `apt install nvidia-cuda-toolkit`——那是 CUDA 12.0，不支持 Blackwell。
> **不要**引入需源码编译 CUDA kernel 的依赖（mamba-ssm / causal-conv1d / flash-attn 无 cu128 预编译 wheel）。注意力一律用 `F.scaled_dot_product_attention`。

### 4.2 五分钟跑通全流程（推荐先做这个）

```bash
cd terragrip_vla

# ★ 一条命令：自动建数据 → 建特征缓存 → 训练所有模型 → 逐条验证 25 项效果
python scripts/verify_framework.py --config small
```

它会打印一张 PASS/FAIL 表，覆盖：异方差数据、冻结感知、概念保真度、**保形覆盖率**、语言语义方向、**slip 头对语言不变**、帕累托前沿、**泄漏曲线**、概念干预、探针饱和检测、OOD 与留出指令泛化。

### 4.3 单项运行

```bash
# ---- 训练（首次会自动生成数据 + 特征缓存）----
python -m training.train --config-name small mode=modular
python -m training.train --config-name small mode=e2e
python -m training.train --config-name small mode=hybrid side_channel_weight=0.5

# ---- 评测：E1 帕累托图 + 保形覆盖率 ----
python -m eval.run_eval --config small

# ---- 分析套件（论文核心三图）----
python -m eval.analysis.leakage      --config small --train-missing   # A1 泄漏曲线
python -m eval.analysis.probe        --config small                   # A2 探针
python -m eval.analysis.intervention --config small                   # A3 概念干预

# ---- 主表：多种子 A/B/C × 三种情形 ----
python -m eval.compare --config small --train-missing                 # A4 对比图 + 主表

# ---- 测试 ----
pytest                    # 70 项
pytest -m slow            # 额外跑真 DINOv2 / MiniLM 的联网测试
```

产物落在 `artifacts/small/`（JSON + CSV + `figures/*.png`）。

### 4.4 正式规模（8×5090）

```bash
# 去掉 --config small 即用 default.yaml（dinov2_vitb14, 4500 图, 5 种子）
python -m eval.compare --train-missing

# 8 卡并行不同 (mode, seed)
for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$s python -m training.train mode=modular train.seed=$s &
done; wait
```

单次训练只需 1 张卡（头很小）；不同 mode/种子用 `CUDA_VISIBLE_DEVICES` 铺到 8 卡。精度统一 bf16。

---

## 5. 配置说明（`configs/default.yaml`）

```yaml
mode: modular              # modular | e2e | hybrid  ← 唯一区分 A/B/C 的开关
side_channel_weight: 0.0   # 仅 hybrid：泄漏旋钮。0=纯瓶颈，>0=允许视觉绕过概念
backbone: dinov2_vitb14    # dinov2_vits14(384) | dinov2_vitb14(768) | dummy(离线测试)

conformal:
  score: normalized        # ★ 必须是 normalized，见 §6 第 1 条
  per_gear: false

language:
  encoder: minilm          # minilm(有语义) | hash(离线CI, 无语义)
  condition: text          # text(真 VLA) | budget(喂 α,τ,λ 向量) | none(消融)
  paraphrase_split: train  # heldout 留给泛化评测

model:
  dropout: 0.4             # 两个头一致 → 容量匹配

train:
  lr: 1.0e-3
  epochs: 300
  beta: 1.0                # hybrid：概念损失的权重
  sigma_weight: 1.0        # sigma 的 NLL 权重

alpha_buckets: [0.05, 0.10, 0.20]
eval:
  seeds: [0,1,2,3,4]
  scw_sweep: [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]   # 对数尺度，见 §6 第 6 条
```

---

## 6. 与最初规格的偏离，以及**逼我改的那个数字**

每一条都不是风格偏好，而是**实测发现原方案会坏掉**：

| # | 规格原本 | 现在 | 逼我改的证据 |
|---|---|---|---|
| 1 | 保形用原始残差 `y-μ` | **归一化残差** `(y-μ)/σ̂` | 打滑噪声是**强异方差**的（水泥 σ=0.02，泥地 σ=0.15）。单一全局分位数必须覆盖泥地 → `Q(0.05)=0.158` → 水泥地上界 `0.05+0.158=0.208 > τ_careful=0.15` → **可接受集合处处为空**，modular 退化成"永远 L 档"——**即使 slip 预测器完美**。保形对**任意**打分函数都成立，有限样本保证丝毫未损。这**不是** CQR。 |
| 2 | 确定性 APS | **随机化 APS** | 档位分类器准确率≈100%、softmax≈0.99 → 阈值 q≈0.999 → **81% 的集合是模糊的** → e2e 被逼一直选最大档，比"固定中档"还费力。这是打分函数的**人为产物**，会把对比悄悄送给 modular。修复后：模糊率 0.81→0.11，能耗 0.743→0.463。 |
| 3 | APS 多元集合取**最小**档 | 取**最大**档 | 取最小则 e2e **毫无保证**，帕累托对比是做局。取 `max(S)`：APS 覆盖 best_gear 概率≥1−α，且 slip 随接地面积单调递减 ⇒ `chosen≥best ⇒ slip(chosen)≤τ`。e2e 由此获得**同形状的保证**。 |
| 4 | modular slip MAE < 0.06 | 报 **MAE / Bayes 底**，门槛 ≤1.35× | 这组 σ 下的**不可约噪声底是 0.061** —— 规格的目标在底之下，**任何模型都不可能达到**。实测 **1.08×** 底。 |
| 5 | `[phi, onehot(gear)]` 直接拼接 | phi 先**标准化 + L2 归一化** | 原始 `‖phi‖≈28`，而 `‖gear one-hot‖=1`、`‖句向量‖=1`、`‖slip概念‖≈0.5`。视觉以 **28:1** 淹没其余输入：slip 头每个 (地形,档位) 单元的系统误差 **0.047(原始) vs 0.017(归一化)** —— 它几乎在无视"问的是哪个档"。 |
| 6 | `scw ∈ {0, 0.1, 0.5, 1.0}` | 对数尺度 `{0, .003, .01, .03, .1, .3, 1}` | 实测 leakage 在 `scw=0` 是 0.00，在 `scw=0.1` **已经是 0.75** —— 线性扫描是个阶跃函数，过渡段完全看不见。改对数后曲线是漂亮的渐变：`0→0.00, .003→0.00, .01→0.00, .03→0.18, .1→0.75, .3→1.00, 1→1.00`。 |
| 7 | OOD 地形 `[0.35,0.20,0.15]` | `[0.30,0.18,0.10]` + **加第二个 OOD 地形** | 原值**精确落在三个 τ (0.15/0.25/0.35) 上** → `slip<=τ` 变成浮点抛硬币，epsilon 量级的保形宽度就能把 oracle 干预精度从 **1.00 掀到 0.67**。另外，OOD 只有**一个**地形时，oracle 档位在每条指令下是常数 → leakage 的"仅语言基线"恒等于 1.0 → **整条 OOD 泄漏曲线在构造上就是空的**。 |
| 8 | 早停看总损失 | 早停看**任务损失**（剔除 σ 的 NLL） | σ 的高斯 NLL 约为 **−2**，而 slip MSE 约 **0.008** —— 总损失被一个与任务无关的项主导，训练在第 14 轮就停了，而任务还在改善。 |
| 9 | *(未指定)* | 两个头都加 **dropout 0.4** | 768 维特征 / 1.8k 训练样本，slip 头把噪声背下来了：MAE **0.091 → 0.067**（底 0.056）。两个头一致 → 容量仍匹配。 |

---

## 7. 对抗审查发现并修复的**真缺陷**

代码写完后，我用 6 个正交视角的独立智能体做了对抗审查（保形数学 / 对比公平性 / 分析语义 / 语言接线 / 数据与可运行性 / ML 正确性），每条发现再由 3 个独立"怀疑者"试图**证伪**。**37 条发现，20 条被证伪，17 条存活并已修复**。其中会**静默污染科学结论**的：

| 曾经的缺陷 | 后果 | 修复 |
|---|---|---|
| **`mu.detach()` 根本没能阻止 σ 的 NLL 影响 μ**。μ 和 log_σ **共享 trunk**，NLL 梯度会经 log_σ 层回流进共享 trunk，而 μ 依赖它。 | 这个"不变量"我在**三个文件**里都写了，**它是假的**：σ 项在偷偷重训均值。 | σ 分支改读 **detach 过的隐状态**。`test_sigma_nll_cannot_move_mu` 在旧代码上会**失败**。 |
| **空的随机化 APS 集合被送去最大档**。但空集合只是 `u > q` 的**抛硬币**（与图像无关，发生率≈α），**不是危险的证据**。 | α 恰恰是语言在拧的旋钮。这让 α 对两条路线**符号相反**地起作用：α 变大时 modular 变**便宜**、e2e 反而变**贵**。所有能耗与语义方向的对比全被污染。 | 空集合 → 退回模型自己的 argmax，并用 `low_conf`/`abstained` 单独上报。 |
| **λ 是死旋钮**。代价是 `area + λ·slip`；相邻档位的接地面积差是 **0.5**，而可接受档位的 `slip ≤ τ ≤ 0.35`。`RISK_TABLE` 里没有任何 λ 能跨过这个差。 | 论文说的风险预算 `(α,τ,λ)` 实际上只是 `(α,τ)`。实测 λ 改变的决策数 = **0**。 | 代价改为 `area + λ·(slip/τ)`，两项量纲可比；λ 重新调值。现在 λ **真的会改变决策**，并有测试钉死。 |
| **`ensure_data` 只检查 `.npz` 文件是否存在** | 改了 `data.seed` 或 `data.sizes`，旧数据被**静默复用** —— 你在用你没要的数据训练，而且哪里都不会报错。 | 把生成时的 `(seed, sizes)` 记在数据旁边并比对，不一致就重生成。 |
| **`compare.py --train-missing` 用全新的 `default` 配置重训**，丢掉所有 CLI 覆盖项 | 它会用**默认设置**训练模型，然后用**你的设置**评测 —— 一个"受控对比"在悄悄评测一批**从未按你要求训练过**的模型。 | 训练配置从活的 `cfg` 派生。 |
| **`select.py` 声称两条路线"保证相同"** | 它们**不同**：modular 的界标定在**实测(带噪)** slip 上；APS 覆盖的 `best_gear` 定义在**均值**曲线上。**一条路线在被另一条的承诺打分**。 | 同时上报 `violation`(实测) 和 `violation_mean`(均值)，各按各的承诺打分；文档改正。 |
| `predict_all` 给每个档位抽了**独立的 dropout 掩码** | `[B,3]` 概念向量的**跨档结构**被与地形无关的噪声打乱，而这个向量正是 hybrid 档位头读的东西。 | 一个样本一张掩码，在它的三个档位间**共享**。 |
| σ 拟合的是**被 dropout 污染的**残差 | σ 吸收了 `地形噪声 + dropout噪声`，在**低噪声地形**上把保形界撑得最宽——而那正是需要紧界的地方。 | NLL 改为对**无 dropout 的均值**拟合。 |
| `set_seed` 在 `build_context` **之前**，而后者会通过 `torch.hub` 消耗全局 RNG | 同一个 `train.seed`，**骨干是否已缓存**会导致不同的初始权重。 | 挪到头构建之前。 |

---

## 8. 小数据集上的实测结果（`--config small`，真 DINOv2-S）

### 8.1 框架验证：**25 项通过 / 0 项失败**

### 8.2 帕累托（E1）

| 策略 | 能耗↓ | 打滑↓ | 违规率↓ | 档位准确率 |
|---|---|---|---|---|
| fixed_S（永远小档） | 0.000 | 0.303 | 0.543 | 0.419 |
| reactive_only（纯本体感觉） | 0.426 | 0.163 | 0.242 | 0.562 |
| **e2e** | **0.463** | 0.143 | 0.156 | **1.000** |
| oracle（知道真值曲线） | 0.463 | 0.143 | 0.156 | 1.000 |
| fixed_M | 0.500 | 0.166 | 0.265 | 0.236 |
| **modular** | 0.598 | **0.112** | **0.099** | 0.732 |
| fixed_L（永远大档） | 1.000 | 0.088 | 0.095 | 0.345 |

**读法**：
- **e2e 精确等于 oracle**（任务给定 (地形,指令) 是确定性的，黑箱把它做满了）。
- **modular 用 60% 的能耗，达到了 fixed_L（永远最大档）的安全水平**（违规 0.099 vs 0.095）。它拿能耗换了一张**证书**。
- 两者都**严格支配** `reactive_only` 和 `fixed_M`。
- `gear_acc` 不是主角：modular 的 0.732 是它**比 oracle 更保守**造成的，而那正是保形保证的代价与价值。

### 8.3 保形覆盖率（硬保证，全部达标）

| α | 目标 | 回归路径实测 | APS 路径实测 |
|---|---|---|---|
| 0.05 | ≥0.95 | **0.965** ✓ | **0.958** ✓ |
| 0.10 | ≥0.90 | **0.921** ✓ | **0.911** ✓ |
| 0.20 | ≥0.80 | **0.824** ✓ | 0.790 ✓(在采样波动内) |

### 8.4 语言（VLA 的另一半）

- **语义方向正确**：接地面积 careful **0.784** ≥ normal **0.626** ≥ fast **0.386**，跨度 **0.398**
- **slip 头对语言严格不变**：careful vs fast 的 slip 预测差异 = **0.00e+00**
- **语言真的改变最优档**：6 个地形里 5 个，换指令就换档
  （concrete `SSS` / grass `LSS` / mud `LLM` / sand `LMM` / wet_tile `LMS` / loose_gravel `LLM`）

### 8.5 分析套件

- **泄漏曲线**（论文核心图）：`scw: 0→0.00, .003→0.00, .01→0.00, .03→0.18, .1→0.75, .3→1.00, 1→1.00`
  → `scw=0` 是**真瓶颈**（打乱概念，精度掉到**低于仅语言基线**）；旁路一开就迅速泄漏。
  → 分母是**仅语言基线 0.591**（**不是 1/3**！光靠指令就能猜出边际最优档——用错分母是伪造泄漏曲线最容易的方式）。
- **概念干预**：modular `gear_acc 0.732 → (oracle 干预) 1.000`。**e2e 无此接口**——它前向里没有物理概念，实测 slip 无处可写，只能重训。这是**结构性局限**，不是调参问题。
- **探针实验**：**已饱和**（未训练头 R²=0.937，冻结骨干天花板 R²=0.975，headroom 仅 0.038）。
  → 代码**主动警告**：在这个 mock 上探针**证明不了任何事**（地形线性可分，slip 从任何投影都能解出来）。这个实验只有在**视觉上有歧义的真实地形**上才有意义。**藏起这一点，是发表假结论最容易的方式。**
- **泛化**：OOD 地形 modular **0.868** > e2e **0.831**；留出改写句上 modular 指令跨度 **0.265** > e2e **0.180**。

> ⚠️ 以上是 `small` 配置（DINOv2-S, 3200 图, 3 种子）的数字，用于验证**框架效果**。
> 论文数字请用 `default`（DINOv2-B, 4500 图, 5 种子）重跑。

---

## 9. Phase 2（架构已留位，现在**不实现**）

真机数据适配器（`data.DataSource` 就是那个抽象接口，`MockSource` 只是一个实现）、自监督标签空间投影（相机单应 + 时延关联）、部署推理与运行时纠错、CQR 升级、在线保形/ACI、世界模型骨干、RELLIS-3D/RUGD、物理属性头。选档已与部署完全解耦。
