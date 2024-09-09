# VQ-log

## Encoder

**befor Encoder：**

信号标准化

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

|      | batch |  lr  | embedding_dim | embedding_num | iteration |  BN  |  MLP   |  En-De  | loss             |      perplexity       |
| ---- | ----- | :--: | :-----------: | :-----------: | :-------: | :--: | :----: | :-----: | ---------------- | :-------------------: |
| Seq1 | 64    | 1e-3 |      512      |      256      |   10000   |  no  | +relu  | s4d-all | 震荡严重         |     最高60～⬆️⬇️⬆️⬇️      |
| Seq2 | 64(   | 1e-3 |      512      |      512      |   12000   |  no  | +relu  | s4d-all | 震荡严重         | 最高110～(3000轮)⬆️⬇️⬆️⬇️ |
| Seq3 | 128   | 1e-3 |      512      |      256      |   8000    | yes  | linear | s4d-all | 前期震荡明显改善 |    非常低同时⬆️⬇️⬆️⬇️     |
| Seq4 |       |      |               |               |           |      |        |         |                  |                       |
|      |       |      |               |               |           |      |        |         |                  |                       |
|      |       |      |               |               |           |      |        |         |                  |                       |
|      |       |      |               |               |           |      |        |         |                  |                       |
|      |       |      |               |               |           |      |        |         |                  |                       |
|      |       |      |               |               |           |      |        |         |                  |                       |

