# PA-bench 独立数字分身实验包

A–D 使用同一个离线入口。现有 Twin、Mega 和冻结权重只读，D 的两个原始 Parquet 位于包内 local/cross。worker 单独加载模型，不导入主实验 runner、接口密钥或调度脚本。正式运行只向本目录 runs 下的全新时间戳目录落盘；smoke 不训练、不评分汇总、不写预测文件。实现对应 [Notion 实施指南](https://app.notion.com/p/422ee3250ae1817386a9e9d08015b34f)。

## 环境和入口

在 PA-bench 根目录使用 PowerShell。数据侧复用 .mega-venv 的 Python 3.11、pyarrow、numpy、scipy、scikit-learn。冻结模型通过 .retrieval-venv 的 torch/transformers 子进程读取本地权重；全部模型调用只在本机。迁移时可用 --data、--cross、--model-python、--model 指定实际位置。重新获取公开 D 数据使用包内 fetch_d.ps1；本机已准备。数据集版本见 local/cross/REVISION。Twin 与 Mega 使用现有项目快照，正式配置记录输入文件哈希。

```powershell
$py = '.\.mega-venv\Scripts\python.exe'
& $py -B isolated_abcd\cli.py smoke
```

以下是日后正式实施的入口，本次交付只执行 smoke：

```powershell
& $py -B isolated_abcd\cli.py select-reader
& $py -B isolated_abcd\cli.py run A --studies context_effects default_eric --holdout default_eric --reader bm25
& $py -B isolated_abcd\cli.py run B --reader bm25
& $py -B isolated_abcd\cli.py run C --reader bm25
& $py -B isolated_abcd\cli.py run D --source two_armed_bandit --reader bm25
& $py -B isolated_abcd\cli.py run D --source restless_bandit --reader bm25
```

示例的 bm25 是可运行默认值。正式比较先在固定开发用户上运行 select-reader，保存输出，锁定同一 reader 和模型。默认是已有的 roberta-large 掩码模型，对一个掩码位置上的全部候选字母打分并归一化；它是可复现的本地评分基线，性能待实测。候选不可单 token 编码或题面自身超过窗口时停止并记录未完成，避免漏选项。可显式指定本地因果模型快照 --arch causal --model <本地目录>，然后在同一模型上比较 empty、bm25 与 gpg。gpg 是包内的提示式画像加原文实现，未复刻原论文所有任务提示。不同模型的结果单列。与主实验 Nex 的成绩不能直接合表。

## 实验定义

- **A** 从 full_persona 揭示真实早期问答，按 PID 接 Mega。默认将 context_effects 作开发目标，整项 default_eric 留出，按需要添加已审计的单选研究。删除题面重叠历史，开发用户形成题库。以开发参考人的真实目标选项分类错数作为预期损失的统计代理：固定策略贪心最小化各分支总错数，自适应策略在与当前回答一致的分支上选题。信息增益基线最大化参考组的回答熵。固定题序、随机题序、自适应、信息增益各在 0/1/3/5/10 题揭示处读取，同一人的随机序列固定。用相近他人的三题证据作附加对照。默认目标是有共有可选题且能回答十题的人，记录被排除数。该首版不对所有可能问卷和模型推理做穷举优化。
- **B** wave_split 以题号、子项、题面、选项语义配对。两波分开评分。开发用户训练“能否重复”分类器；校准用户选择温度、全局与分层混合权重。独立测试用户的两波答案只在当前预测后由评测器使用。另有明确显示上次同题选择的复测任务。全量 B 会对同一用户的多道题独立打分，需单独安排时间。
- **C** 仅取 context_effects 真正的 Bin、Attr、Comp 单选题，排除练习。提取价格与各产品的主要质量属性，以条件 logit 比较独立效用、共享菜单特征、与个人画像的交互。画像是历史题答的固定哈希投影。最低价、折中位置、相对价格和价格/质量支配为显式菜单特征；直接读取整组菜单的冻结模型、打乱参数、相近他人画像作为对照。支持实际展示预测；不可据此推断同人同商品反事实。
- **D** 使用 Cross-Psych-101 官方 train 中开发用户估计共享学习率，官方 test 中按用户哈希留出的参与者做最终预测，重复目标轨迹按人去重。来源可选四臂 restless_bandit、correlated_bandit 或同任务 two_armed_bandit，目标均为 safe/risky。来源行为估计个人学习率，目标价值从零初始化，留出目标标签绝不参与拟合。对比复制上次动作、共享、本人、跨人打乱的学习率及读轨迹文本的冻结模型。两种任务固定奖励缩放与惯性。Cross-Centaur 原训练权重属于可另行复现的外部强基线，本包未内置其成绩或模型。

## 数据审计与结果

默认 dev/cal/test 由 SHA-256(PID) 取模 100 固定为 60/20/20。发布的同一人可跨研究；任何拟合都在指定的人群与研究上完成。读取时只接受答案与候选语义可映射的单选和单选矩阵行；题型映射、重复键、输入或概率异常不会静默生成成绩。正式输出为 config、inputs、splits、labels、predictions、summary、status。预测器只接收选项、题面与该条件可见的历史；标签与概率分开保存。若状态 incomplete，检查错误，不把部分输出当作正式结果。

summary 中 label1/label2 分别为同一预测对应的各个评分标签，B 的 seen 协议用单独方法名。报告用户宏平均精确一致率、NLL、Brier，以及有同样样本的预设参考法上的每百题净改善、改正/改错数及 2,000 次用户配对 bootstrap 区间。零概率的对数损失以 1e-12 截断。输入日志可能包含公开调查记录，留在本地。实际阶段预留 held-out 研究、排除计数、模型上下文上限和目标样本量需在开发数据上锁定后再使用正式入口。

## 连通检查边界

smoke 只走一组三题历史和真实 Mega 菜单、一个 Twin 两波配对、一个菜单设计矩阵组、两条 D 来源/目标记录与本地模型调用。不会拟合 A 的题库策略、B 的校准器、C 的菜单参数或 D 的群体学习率，也不计算任何研究成绩。首轮完整运行的耗时、收敛、整体样本保留率和方法效果均待正式验证。

