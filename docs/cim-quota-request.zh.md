# 光子相干依辛机（CIM）算力申请文案

可直接提交至 [platform.qboson.com](https://platform.qboson.com/) / 开物量子活动中心或项目对接人。

---

## 申请摘要（短版）

申请 **QBoson 光子相干依辛机（CIM / SPQC）真机配额**，用于 scRNA-seq QUBO 基因面板的**硬件对照**。  
本地 CPU（Ocean / Kaiwu **tabu**）仅负责 α 搜索与事后精修；**依辛最小化本身必须在云端 CIM 上完成**，不能用经典 tabu 替代。本地提交端**不需要 GPU**。

- **真机任务量**：约 **5–10 次**（v1/v2 各 1 次正式 + 2–3 次重复）
- **问题规模**：$n\sim 5\times 10^3$ 稠密 Ising（8-bit 系数；PrecisionReducer 后可能略增变量）
- **模式**：优先 **optimization（原 Quota）**；必要时 sampling，64–128 samples/任务

---

## 正式申请正文

**项目名称**  
单细胞转录组 QUBO 特征选择的光子依辛机（CIM）对照求解

**申请资源类型**  
相干光量子计算云平台 **CIM / SPQC 真机**（Kaiwu `CIMOptimizer`），**非**本地经典 tabu 配额。

**科学目的**  
在已锁定的同一 QUBO 矩阵 $Q(\alpha^*)$（Romero et al. Eq. 4；约 5,000 基因、$k\approx 50$）上，用光子 CIM 求解依辛形式，与已完成的 Ocean tabu / Kaiwu 本地 tabu 结果比较：**能量 $F^\top Q F$、|F\*|、基因成员重叠**。  
目标是完成**光子硬件对照**，不是再发现 $k$，也不是用 CIM 打 LASSO 的 Ridge MSE。

**为何必须申请 CIM（不能只用 CPU tabu）**

| 阶段 | 后端 | 是否占 CIM 配额 |
| --- | --- | --- |
| α 二分，锁定 $Q(\alpha^*)$ | 本地经典 CPU：Ocean / Kaiwu **tabu** | 否 |
| 同一 $Q$ 的依辛最小化（对照核心） | **云端光子 CIM** | **是（本次申请）** |
| 解码后 1-bit 精修 | 本地 CPU（精确 $Q$） | 否 |

说明：

1. 经典 tabu 对照（Ocean∩Kaiwu）**已经完成**，不消耗真机配额。  
2. 本申请专用于 **photonic CIM 对照**；若无真机配额，则无法完成该硬件路径。  
3. “本地无 GPU”仅指提交客户端与经典预处理；**不表示无 CIM 需求**。

**资源与次数估算**

- 数据集：GSE308682（v1）与胎儿造血 Smart-seq2（v2），可各跑一轮。  
- 每队列：**1 次正式提交 + 2–3 次重复**（换 `task_name` 估成员抖动 / 调采样与精度兜底）。  
- **合计约 5–10 次真机任务**。  
- 不做 α 全流程上机（避免约 17 次无效探测浪费配额）。  
- 单任务规模：全连接 Ising，$n\approx 5000$（+辅助自旋）；优化模式由平台内部分配千次级演化；若用采样模式，建议 64–128 samples/任务（上限 2000）。  
- 系数精度：按平台 **8-bit**（$[-128,127]$），经 SDK `PrecisionReducer` 适配。

**验收标准（科学）**

- 返回稀疏掩码，能量 $E<0$，|F\*| 接近 $k=50$；  
- 与经典 tabu 比较能量与成员；v2 期望仍覆盖关键干性基因（如 MLLT3 / HOPX / SPINK2 / NPR3）；  
- 巨大 |F\*| 或 $E\gg 0$ 记为该规模下的工程/物理约束，仍属有价值的真机结果。

**本地环境（不申请 GPU）**

- Python 3.10 + Kaiwu SDK；普通 CPU 即可完成 MI、α 搜索、提交与精修。  
- 真机算力全部在平台 CIM/SPQC 侧。

**联系与实现**  
代码路径：`SOLVER="tabu"` + `COMPARE_KAIWU_CIM=True`（或 `SOLVER="kaiwu_cim"`，α 仍默认经典）。仓库说明见 `README.md`「CIM / Ising machine」与本文件。

---

## 一句话版（表单字数紧时）

> 申请 CIM/SPQC 真机约 5–10 次任务，对已锁定的 $n\sim5000$ 稠密 Ising（QUBO 基因面板）做光子硬件对照；α 搜索用本地 CPU tabu，依辛求解必须上 CIM，本地无需 GPU。
