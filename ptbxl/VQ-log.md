# VQ-log

## Encoder

**befor Encoder：**

信号标准化：

高斯噪声：

Mask：

**with Encoder：**

S4D：





**after Encoder：**

BN + MLP



## Decoder

**befor Decoder：**

mlp

**with Decoder：**

S4D：





**after Decoder：**

BN + MLP





## codebook（Quantizer）

hyper para：batch，lr，embedding_dim，embedding_num

problem：需不需要新的loss传播策略，**codebook的大小**，

------

**lr：**

学习率大概在1e-5的时候明显感觉codebook的perplexity提升很小，目前感觉学习率在1e-3～1e-4之间（不绝对）

**embedding：**

数量多的codebook





## loss

**Task1:**

针对重建的任务：MSE，commitmen loss

情况：loss震荡

可能存在的问题：数据量不足导致学出来的信号只有明确的波峰波谷，或者考虑生成低分辨率的信号。

------

**Task2:**







## Experiment

table：s4d本身的工作效果









table：超参调整

(based on train，test on valid)

|                         | batch |  lr  | emb_dim | emb_num | iteration |  BN  |       MLP        |  En-De  |                  loss    趋势                   |                     perplexity                      | commit-loss |            trick            |
| :---------------------: | :---: | :--: | :-----: | :-----: | :-------: | :--: | :--------------: | :-----: | :---------------------------------------------: | :-------------------------------------------------: | :---------: | :-------------------------: |
|          Seq1           |  64   | 1e-3 |   512   |   256   |   10000   |  no  |      +relu       | s4d-all |                    震荡严重                     |                    最高60～⬆️⬇️⬆️⬇️                     |    0.25     |                             |
|          Seq2           |  64(  | 1e-3 |   512   |   512   |   12000   |  no  |      +relu       | s4d-all |                    震荡严重                     |                最高110～(3000轮)⬆️⬇️⬆️⬇️                |    0.25     |                             |
|          Seq3           |  128  | 1e-3 |   512   |   256   |   8000    | yes  |      linear      | s4d-all |                前期震荡明显改善                 |                   非常低同时⬆️⬇️⬆️⬇️                    |    0.25     |                             |
|       优化VQ之后        |       |      |         |         |           |      |                  |         |                                                 |                                                     |             |                             |
|          Seq4           |  128  | 1e-3 |  2048   |   256   |   12000   | yes  |      linear      | s4d-all |                后期震荡相对麻烦                 | 从1160开始逐步下降，最后下降到160左右，还有下降趋势 |      1      |                             |
|       Seq5-model2       |  128  | 1e-3 |  2048   |   256   |   1600    | yes  |      linear      | s4d-all |                   还是有震荡                    |                      1170->420                      |             |                             |
|          Seq6           |  128  | 5e-4 |  1024   |   256   |   12000   | yes  |      linear      | s4d-all |                                                 |                                                     |             |                             |
|          Seq7           |  128  | 1e-4 |  1024   |   256   |  100000   | yes  |      linear      | s4d-all |                                                 |                                                     |             |                             |
|       增大数据集        |       |      |         |         |           |      |                  |         |                                                 |                                                     |             |                             |
|          Seq8           |  64   | 1e-3 |  1024   |   128   |   10000   | yes  |      linear      | s4d-all | 基本消除震荡，后期略微震荡，loss最后在0.220左右 |                从八百多下降到220左右                |      1      |                             |
|    *Seq9-model-val1     |  64   | 1e-3 |  1024   |   128   |   10000   | yes  |      linear      | s4d-all |           基本消除震荡，后期略微震荡            |                     880->两百多                     |      1      |             SGD             |
|          Seq10          |  64   | 1e-3 |  1024   |   256   |   10000   | yes  |      linear      | s4d-all |    基本消除震荡，后期略微震荡，降低到0.1左右    |                 10000轮降到400左右                  |    0.25     |  SGD+kmeans+lowerdimension  |
|          Seq11          |  64   | 1e-3 |  1024   |   256   |   10000   | yes  |      linear      | s4d-all |   基本消除震荡，后期略微震荡，降低到0.044左右   |                   10000轮降到214                    |    0.25     |         Adam+kmeans         |
|    *Seq12-model-val2    |  64   | 1e-3 |   512   |   256   |   10000   | yes  |      linear      | s4d-all |   基本消除震荡，后期略微震荡，降低到0.05左右    |            10000轮最后**收敛**到165左右             |    0.25     |         Adam+kmeans         |
|    *Seq13-model-val3    |  128  | 1e-3 |   512   |   256   |   10000   | yes  |      linear      | s4d-all |                 收敛到0.045左右                 |            10000轮最后**收敛**到186左右             |    0.25     |         Adam+kmeans         |
|    *Seq14-model-val4    |  64   | 1e-3 |   512   |   256   |   10000   | yes  |    **conv1d**    | s4d-all |                 收敛到0.054左右                 |            10000轮最后**收敛**到197左右             |    0.25     |         Adam+kmeans         |
| **\**Seq15-model-val5** |  128  | 1e-3 |   512   |   256   |   10000   | yes  |      conv1d      | s4d-all |                 收敛到0.057左右                 |            10000轮最后**收敛**到217左右             |    0.25     |         Adam+kmeans         |
|    *Seq15-model-val7    |  64   | 1e-3 |   512   |   256   |   10000   | yes  |      conv1d      | s4d-all |           收敛到0.011左右（loss取和）           |            10000轮最后**收敛**到46*8左右            |    0.25     | multi-head-VQ-8+Adam+kmeans |
|    *Seq15-model-val8    |  64   | 1e-3 |   512   |   256   |   10000   | yes  |  conv1d加深一层  | s4d-all |                 收敛到0.014左右                 |            10000轮最后**收敛**到43*8左右            |    0.25     |     MHVQ-8+Adam+kmeans      |
|   *Seq15-model-val8.1   |  64   | 1e-3 |   512   |   256   |   10000   | yes  | conv1d再下卷一层 | s4d-all |                 收敛到0.014左右                 |            10000轮最后**收敛**到567左右             |    0.25     |     MHVQ-8+Adam+kmeans      |
| **\**Seq15-model-val9** |  128  | 1e-3 |  1024   |   256   |   15000   | yes  |  conv1d加深一层  | s4d-all |                 收敛到0.009左右                 |           15000轮最后**收敛**到102*8左右            |    0.25     |     MHVQ-8+Adam+kmeans      |
|   *Seq15-model-val10    |  256  | 1e-3 |  1024   |   256   |   10000   | yes  |  conv1d加深一层  | s4d-all |                 收敛到0.013左右                 |            10000轮最后**收敛**到791左右             |    0.25     |     MHVQ-8+Adam+kmeans      |
|                         |       |      |         |         |           |      |                  |         |                                                 |                                                     |             |                             |
|                         |       |      |         |         |           |      |                  |         |                                                 |                                                     |             |                             |
|                         |       |      |         |         |           |      |                  |         |                                                 |                                                     |             |                             |
|                         |       |      |         |         |           |      |                  |         |                                                 |                                                     |             |                             |
|                         |       |      |         |         |           |      |                  |         |                                                 |                                                     |             |                             |





**trick：**

data: gaussian, mask

VQ: kmeans☑️, 低维表征(2800->350)☑️, residual☑️, group, FSQ

Decoder: hierarchy, multi-projection, 

batch大☑️☑️（batch越大codebook使用率越高）

encoder->VQ->decoder平滑过渡☑️（多层卷积+反卷积）

decoder->projection过渡更平滑一点



encoder/decoder选择？

下游任务





更换decoder部分的sequential位置

**在ptbxl数据集上的表征，finetune**

表征的特征维度是否可以再低一点？

111 codebook是否可以更大一点？（对罕见病的表征怎么样）

如何增强表征的丰富性？感知loss





| Finetune                                      | batch | lr   | emb_dim | emb_num | iteration | BN   | MLP        | En-De   | loss    趋势              | perplexity                  | commit-loss | trick              |
| --------------------------------------------- | ----- | ---- | ------- | ------- | --------- | ---- | ---------- | ------- | ------------------------- | --------------------------- | ----------- | ------------------ |
| Based on model-val-5->model-finetune-256      | 256   | 1e-3 | 256     | 512     | 10000     | Yes  | Conv1d     | s4d-all | 有点震荡，最后收敛到0.112 | 稍有震荡，最后稳定到194左右 | 0.25        | Adam+kmeans        |
| Based on model-val-9->model-finetune-256-1024 | 256   | 1e-3 | 256     | 1024    | 10000     | Yes  | conv1d加深 | s4d-all | 收敛到0.028左右           | 先下降后上升到733           | 0.25        | MHVQ-8+Adam+kmeans |
|                                               |       |      |         |         |           |      |            |         |                           |                             |             |                    |
|                                               |       |      |         |         |           |      |            |         |                           |                             |             |                    |
|                                               |       |      |         |         |           |      |            |         |                           |                             |             |                    |







| GPT-gen  | vocab=emb-num | block-size | iter  | default    | lr   | Batch | given-seq | Trick | Result                                                       |
| -------- | ------------- | ---------- | ----- | ---------- | ---- | ----- | --------- | ----- | ------------------------------------------------------------ |
| GPT-test | 1024          | 32 * 8     | 20000 | gopher-44m | 5e-4 | 64    | 64 * 8    |       | Loss:2.2，loss稍有震荡，生成的序列学到了节律，但整体精度不行 |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |
|          |               |            |       |            |      |       |           |       |                                                              |

