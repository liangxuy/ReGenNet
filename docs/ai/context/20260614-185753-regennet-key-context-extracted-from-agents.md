# ReGenNet 关键上下文提取

本文从 2026-06-14 的根目录 `AGENTS.md` 提取长期项目记忆。根 `AGENTS.md` 后续只保留极简入口，详细阶段记录仍以 `docs/ai/context/` 下各时间戳文档为准。

## 工作约束

- 默认使用中文沟通、写文档、写计划和总结；代码注释只写原因，不写过程。
- 新增上下文、设计、取舍和结果记录统一放到 `docs/ai/context/`，使用 `YYYYMMDD-HHMMSS-文件名.md` 命名。
- 新增上下文文件时只创建新文件，不覆写、重命名或删除历史上下文文件。
- 进入实现前应先完成 design / plan，并把计划落盘到 `docs/ai/context/`。

## 当前论文主线

- 主线已从复现 ReGenNet Table 4 转为 `Interaction-aware joint forecasting of two-person human motion from partial observations`。
- 中文表述：基于部分观测的交互感知双人动作联合未来预测。
- 第一阶段固定 InterHuman：150 帧窗口，前 30 帧观测，后 120 帧预测。
- `前 20% / 后 80%` 作用在固定 150 帧窗口上，不直接作用在原始全长序列；过短序列过滤，不用 padding 伪造未来。
- 核心比较对象：repeat / zero-velocity、independent predictor、concat no-relation predictor、relation-aware predictor。
- 关键指标：MSE、rotation MSE、translation MSE、relative root distance error、relative orientation error、inter-person distance consistency、long-horizon error。

## 论文结论边界

- 不能声称 multi-person forecasting、interaction-aware forecasting、explicit relation modeling 或 long-horizon forecasting 首创。
- 当前可成立的贡献应收窄为：InterHuman SMPL 150/30/120 deterministic two-person forecasting protocol、SMPL active-vector 数据/评估闭环、严格同口径 empirical study，以及轻量 root-level relation cues 相对 concat / parameter-matched concat 的稳定收益。
- relation-aware 不优于 independent 的 `future_mse/long_mse`，也不优于 repeat 的部分 relation-style metrics；论文不能写成全面最优。
- P6 qualitative 只能解释 P5 全 test aggregate，不能替代主表证据。

## InterHuman-AS / ReGenNet 历史线

- InterHuman-AS Table 4 主指标是 `FID / Acc. / Div. / Multimod.`，不是 `R Precision / MM Dist`。
- Table 4 主路径应使用 ST-GCN recognition feature evaluator；text-motion evaluator 只作为扩展。
- 本地 InterHuman 数据没有 text/caption 目录，也没有 InterHuman recognition checkpoint。
- 当前 InterHuman-AS 实现是 SMPL reproduction attempt；若要精确对齐论文 SMPL-X 口径，需要补 SMPL-X 转换或确认官方 evaluator。
- 已生成冻结 H5：`dataset/interhuman/smpl/conditioned/interhuman_{train,val,test}.h5`。
- ReGenNet InterHuman 50K baseline 已按用户要求停止，最后可用 checkpoint 为 `save/interhuman/paper_config_l8_d512_accum64_50000_baseline/model000020000.pt`。

## Forecasting P1-P6 状态

- P1 已完成 InterHuman forecasting dataset、active vector extract/restore、150 帧裁剪、normalizer 和 shape/finite smoke。
- P1 smoke：train/val/test 可用样本 `2910/226/508`，batch shape `obs=[4,30,2,147]`、`target=[4,120,2,147]`。
- P2 已完成 original-scale metrics、metrics sanity 和 repeat baseline evaluator。
- P2 repeat test：`future_mse=0.0368928675`，`long_mse=0.0511287494`，`relative_root_distance_error=0.2552213891`。
- P3 已完成 independent / concat baseline 训练和 checkpoint evaluation。
- P3 seed0：independent `future_mse=0.0287435004,long_mse=0.0361207679`；concat `future_mse=0.0319019718,long_mse=0.0378956974`。
- P4 已完成 relation-aware predictor；seed0 relation `future_mse=0.0314433519,long_mse=0.0369622079,relative_root_distance_error=0.4089161090`。
- P5.2 主表 3-seed 已完成：relation 相对 concat 的 `long_mse` mean 更低且 same-seed 3/3 胜出，`relative_root_distance_error` mean 更低且 same-seed 3/3 胜出。
- P5.2 mean：repeat `future_mse=0.0368928675,long_mse=0.0511287494`；independent `future_mse=0.0287863306,long_mse=0.0362148034`；concat `future_mse=0.0320573101,long_mse=0.0380507699`；relation `future_mse=0.0317788706,long_mse=0.0373418675`。
- P5.3 消融 3-seed 已完成：all-features relation 优于 parameter-matched concat；encoder 收益存在但幅度小；velocity-only 与 all-features 接近，需保留边界。
- P6 qualitative 已完成，输出在 `results/forecasting/interhuman/p6_qualitative_150_30_120/`，样本覆盖 success / close / failure / boundary。
- P1-P6 总体状态审计结论：InterHuman 150/30/120 第一阶段工程证据链完成；论文正文、投稿材料、P5.4 observation-ratio、P7/P8 数据集扩展和动作视频渲染尚未完成。

## Related Work 关键定位

- 已重点调研 SoMoFormer、Trajectory2Pose、MRT、JRT、DuMMF / Stochastic Multi-Person 3D Motion Forecasting。
- SoMoFormer：joint-coordinate trajectory token，不是 timestep token；每个 token 内部仍包含 obs+future padding 的时间轨迹。
- T2P：先预测多模态 global hip trajectory，再用 trajectory conditioning 生成 local pose；不能被用来支持当前项目的首创性。
- JRT：显式 joint-to-joint relation，包括 relative distance、bone adjacency、same-body connectivity、relation-aware attention 和未来 inter-joint distance supervision。
- 推荐大白话定位：MRT = 看自己 + 看别人；SoMoFormer = 关节轨迹 token 互相看；JRT = 给关节 attention 加人体关系地图；DuMMF = 预测多个合理未来；T2P = 先预测人往哪走，再补身体怎么动。

## P7 / P8 SoMoFormer 迁入状态

- 用户已确认最终目标是完整迁入官方 SoMoFormer 架构，而不是继续维护 lite 版。
- P7.1 joint-space SoMoFormer baseline 已实现并 smoke 通过，新增 `utils/forecasting_xyz.py`、`model/forecasting_somoformer.py`、`train/train_forecasting_xyz.py`、`eval/eval_forecasting_xyz.py`。
- P7 `independent_pair_xyz` 定义：仍使用双人样本，但模型内部把两个人分开独立预测，最后拼回双人输出。
- P7 xyz 3-seed mean：`independent_pair_xyz joint_mse=0.0682312447,mpjpe=0.3375963565,long_joint_mse=0.1284009837,relative_root_distance_error=0.2280735768`；`somoformer_xyz joint_mse=0.0607606915,mpjpe=0.2947844890,long_joint_mse=0.1186761773,relative_root_distance_error=0.2032881472`。
- P7 结论只适用于 joint-space 口径，不能直接横比 P5 active-vector 主表。
- P8 已新增 `official_somoformer_xyz`，保留 `somoformer_xyz` 作为 lite baseline。
- P8 official 迁入 `AuxilliaryEncoder`、`LearnedDoublePositionalEncoding`、官方 DCT matrix 逻辑、`grid|neck|naive` location method、auxiliary loss、padding mask 接口和 train-time `metamask`。
- P8 official 3-seed mean：`joint_mse=0.0706294909,mpjpe=0.3033010067,long_joint_mse=0.1376462394,relative_root_distance_error=0.1935767838,inter_person_distance_consistency_xyz=0.0054475044`。
- P8 结论：official 在 relation metrics 上更好，但 `joint_mse/mpjpe/long_joint_mse` 更差，不能声称完整官方架构全面优于 lite baseline。

## T2P / JRT 源码与数据复现

- T2P 官方源码已克隆到 `/home/rpartx3080/CodeSpace/T2P`，当前 HEAD `25ccf4fbc5a2e4fecdb2aee250201d7d8849ecda`。
- T2P 官方 release 可公开下载数据已保存并解压；JRDB-GMP 完整复现阻塞在 JRDB 原始数据授权下载。
- T2P 官方仓库不是开箱即跑，Hydra config 和硬编码路径需要修复。
- 不建议直接修改 T2P 官方仓库读取 ReGenNet 数据；应在 ReGenNet 内新增 `t2p_interhuman_xyz` adapter/model，复用 InterHuman split 和 P7/P8 xyz evaluator。
- 2026-06-15 用户已确认 T2P 迁移边界：第一版只做 InterHuman 双人 xyz deterministic T2P-style baseline，不使用 JRDB-GMP，不跑官方 preprocessing，不启用 best-of-6，代码落在 ReGenNet 内，主排行只和 `independent_pair_xyz / somoformer_xyz / official_somoformer_xyz` 同口径比较。
- P9 计划已创建：`docs/ai/context/20260615-132220-p9-t2p-interhuman-xyz-plan.md`。后续实现应新增 `t2p_interhuman_xyz`，保留 T2P 的 trajectory-first、global-local decoupling 和 trajectory-conditioned local pose decoder，输出 `[B,120,2,24,3]` 并复用 P7/P8 xyz metrics。
- JRT 官方源码已克隆到 `/home/rpartx3080/CodeSpace/JRTransformer`，当前 HEAD `3765b17cb8b7ba1adfdec42839732fe93a0cebb3`。
- JRT 官方 3DPW 预处理数据和 checkpoint 已下载，官方 test 已跑通：`Test VIM avg=39.51,900ms=68.57`，与论文 Table 1 对齐。
- 若迁入 ReGenNet，建议先做 joint-space `jrt_xyz` baseline，复用 `InterHumanForecastDataset -> active_to_xyz -> train/eval_forecasting_xyz.py`。
- `jrt_xyz` 第一版只建议支持 InterHuman SMPL H5，输入 `[B,30,2,24,3]`，输出 `[B,120,2,24,3]`。

## DuMMF 源码与数据复现

- 2026-06-14 已克隆 DuMMF 官方源码到 `/home/rpartx3080/CodeSpace/DuMMF`，commit 为 `28bf5efc709cdd4dcd5847906920dd8af0b33c48`。
- 可公开直接下载的数据已落地：CMU ASF/AMC、MuPoTS-3D `TS1..TS20/annot.mat`、SoMoF PoseTrack/3DPW JSON。
- DuMMF 公开仓库当前是 diffusion-based 代码，默认训练依赖 AMASS-CMU `.npz` 和 SMPL-H/DMPL body models；这些需要账号/协议手动准备，不能把已下载的 CMU ASF/AMC 误当成 diffusion 训练数据。

## 重要上下文入口

- 最终正式设计：`docs/ai/context/20260603-190003-forecasting-final-official-design.md`
- P1-P6 完整设计 contract：`docs/ai/context/20260603-184214-forecasting-p1-p6-complete-design.md`
- P5 主表结果：`docs/ai/context/20260603-232353-forecasting-p5-main-table-result.md`
- P5 消融结果：`docs/ai/context/20260604-085116-forecasting-p5-ablation-result.md`
- P6 qualitative 结果：`docs/ai/context/20260604-091820-forecasting-p6-qualitative-result.md`
- related work 深度定位：`docs/ai/context/20260609-094222-forecasting-five-paper-deep-review-innovation.md`
- P7 xyz 3-seed 结果：`docs/ai/context/20260614-104849-p7-xyz-main-3seed-gpu-result.md`
- P8 official SoMoFormer 结果：`docs/ai/context/20260614-143029-p8-official-somoformer-xyz-3seed-result.md`
- T2P 复现与适配：`docs/ai/context/20260614-184859-t2p-source-data-reproduction-result.md`、`docs/ai/context/20260614-185128-t2p-on-regennet-dataset-adaptation-design.md`
- JRT 复现与适配：`docs/ai/context/20260614-185253-jrt-source-data-reproduction-result.md`、`docs/ai/context/20260614-185518-jrt-on-regennet-dataset-adaptation-design.md`
- T2P 迁移边界与 P9 计划：`docs/ai/context/20260614-185453-t2p-migration-boundary-confirmation.md`、`docs/ai/context/20260615-132220-p9-t2p-interhuman-xyz-plan.md`
- DuMMF 复现结果：`docs/ai/context/20260614-200410-dummf-source-data-reproduction-result.md`
