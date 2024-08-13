# DDPM

$$
reverse: denosie\\
X_T\rightarrow\dots \rightarrow X_t \rightarrow^{p(X_{t-1}|X_{t})} \rightarrow
X_{t-1} \rightarrow \dots X_0
\\
diffusion: + \epsilon \\
X_T\leftarrow \dots \leftarrow X_t \leftarrow^{q(X_{t}|X_{t-1})} \leftarrow
X_{t-1} \leftarrow \dots X_0
$$



## Diffusion process

$$
diffusion: + \epsilon (noise) \\
X_T\leftarrow \dots \leftarrow X_t \leftarrow^{q(X_{t}|X_{t-1})} \leftarrow
X_{t-1} \leftarrow \dots X_0
$$

diffusion过程重点在于为原始图片增加T步噪声，先规定噪声的形式：
$$
q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_{t};\sqrt{1 - \beta_t}\boldsymbol{x}_{t-1}, \beta_t \boldsymbol{I})
$$
其中$\boldsymbol{\beta}$是一个**随着t增大的单增数列**，取值均为$(0,1)$​​​。

------

diffusion过程最重要的是**如何加噪声？**

**`参数重整化：`**令$\alpha_t = 1 - \beta_t$，我们采样得到加噪后的新样本，采样的$\boldsymbol{\epsilon}$来自标准高斯分布（$\boldsymbol{x}_{0}$是输入）：
$$
\boldsymbol{x}_{t} = \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} + 
\sqrt{\alpha_t}\boldsymbol{x}_{t-1}
$$
（先采样标准高斯然后乘上标准差，再加上均值，这样的话更好参数学习）

继续带入$\boldsymbol{x}_{t-1}$，我们可以得到：

> **Lemma1**
>
> 两个高斯分布$\mathcal{N}_1(\mu_1, \sigma^2_1), \mathcal{N}_2(\mu_2, \sigma^2_2)$叠加后的新分布满足$a\mathcal{N}_1 + b\mathcal{N}_2 = \mathcal{N}_3(a\mu_1+b\mu_2, a^2\sigma^2_1+b^2\sigma^2_2)$。

$$
\boldsymbol{x}_{t} = \sqrt{1 - \alpha_t\alpha_{t-1}}\bar{\boldsymbol{\epsilon}}_{t-2} + 
\sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2}
$$

进一步展开可以得到：
$$
\boldsymbol{x}_{t} = \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} + 
\sqrt{\bar{\alpha}_t}\boldsymbol{x}_{0}
\label{0->t}
$$
其中$\bar{\alpha}_t = \alpha_t\alpha_{t-1}\dots\alpha_{1}$。

所以换句话说，给定**输入**，给定**t的步数**，给定**方差数列**，就可以直接通过上式求得任意t时刻的样本。
$$
q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0}) = \mathcal{N}
(\boldsymbol{x}_{t};\sqrt{\bar{\alpha}_t}\boldsymbol{x}_{0}, (1 - \bar{\alpha}_t)\boldsymbol{I})
$$

------

在给定输入$\boldsymbol{x}_{0}$的情况下，可以直接用$\boldsymbol{x}_{t}$预测$\boldsymbol{x}_{t-1}$，即求$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_{0})$：
$$
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) = 
\mathcal{N}(\boldsymbol{x}_{t-1}; 
\tilde{\mu}_t(\boldsymbol{x}_{t}, \boldsymbol{x}_{0}), \tilde{\beta}_t\boldsymbol{I})
$$
下面求解均值和方差，先使用贝叶斯：
$$
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_{0})
= \frac{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1}, \boldsymbol{x}_{0})
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{0})}
{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})}
= \frac{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{0})}
{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})} (markov-chain)
$$

等式右边分子分母三项均在上文给出高斯分布的均值和方差，直接带入高斯分布的PDF：
$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x - \mu)^2}{2\sigma^2})
$$
我们得到新的表达式：
$$
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) \propto
exp(
\frac{(\boldsymbol{x}_{t} - \sqrt{\alpha_t}\boldsymbol{x}_{t-1})^2}{\beta_t} + 
\frac{(\boldsymbol{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\boldsymbol{x}_{0})^2}{1 - \bar{\alpha}_{t-1}} - 
\frac{(\boldsymbol{x}_{t} - \sqrt{\bar{\alpha}_{t}}\boldsymbol{x}_{0})^2}{1 - \bar{\alpha}_{t}}
)
$$
化简得到结果（右式把$\boldsymbol{x}_{t-1}$当作自变量的函数）：
$$
(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})
\boldsymbol{x}_{t-1}^2 + 
(-\frac{2\sqrt{\alpha_t}\boldsymbol{x}_{t}}{\beta_t} - 
\frac{2\sqrt{\bar{\alpha}_{t-1}}\boldsymbol{x}_{0}}{1 - \bar{\alpha}_{t-1}})
\boldsymbol{x}_{t-1} +
C
$$
根据二次函数(高斯分布的PDF)的展开式：
$$
\tilde{\mu}_t(\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) = 
\frac{(\frac{\sqrt{\alpha_t}\boldsymbol{x}_{t}}{\beta_t} + 
\frac{\sqrt{\bar{\alpha}_{t-1}}\boldsymbol{x}_{0}}{1 - \bar{\alpha}_{t-1}})}
{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})}
=
\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})\boldsymbol{x}_{t}+
\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)\boldsymbol{x}_{0}
}
{1 - \bar{\alpha}_t}
$$

$$
\tilde{\beta}_t = 
\frac{1}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} =
\frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}
$$

对于均值来说，通过公式$\ref{0->t}$替换掉$\boldsymbol{x}_{0}$，我们就可以得到均值只和$\boldsymbol{x}_{t}, \epsilon$​的关系：
$$
\boldsymbol{x}_{0} = 
\frac{1}{\sqrt{\bar{\alpha}_{t}}}(\boldsymbol{x}_{t} - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon})
$$

$$
\tilde{\mu}_t(\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) =
\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})\boldsymbol{x}_{t}+
\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)
\frac{1}{\sqrt{\bar{\alpha}_{t}}}(\boldsymbol{x}_{t} - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon})
}{1 - \bar{\alpha}_t}
$$

$$
\tilde{\mu}_t(\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) = 
\frac{1}{\sqrt{\alpha_t}}(
\boldsymbol{x}_{t} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon
)
$$




## Reverse process

我们的终极目标是最大化后验概率$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})$，我们作出高斯假设：
$$
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}) = 
\mathcal{N}(\boldsymbol{x}_{t-1};\mu_{\theta}(\boldsymbol{x}_{t},t), \Sigma_{\theta}(\boldsymbol{x}_{t},t))
$$
所以在reverse过程中，我们的目标就是获得$\mu_{\theta}, \Sigma_{\theta}$。怎么获得？

**`Loss:`**我们需要设计一个优化方案，即让模型预测什么信息？

- 直接预测$\mu_{\theta}, \Sigma_{\theta}$;
- 直接预测$\boldsymbol{x}_{0}$；
- 直接预测$\epsilon$。（采用）

在此之前，我们需要先完成对MAP的优化，直接MAP不可求解，所以 对于此类生成模型均为计算MLE，我们先构建一个似然函数。

>实际上DDPM本质上做的一件事就是把MLE化简优化成一个MSE，生成模型本来是不存在MSE的。

$$
KL(q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})
\,||\,
p_{\theta}(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0}))
= \mathbb{E}_{q}(\log 
\frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{0:T})/ p_{\theta}(\boldsymbol{x}_{0})})
= \mathbb{E}_{q}(\log 
\frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{0:T})} + \log p_{\theta}(\boldsymbol{x}_{0}))
$$

同时对于KL散度有$KL(q||p) \geq 0$，所以有：
$$
-\mathbb{E}_q(\log p_{\theta}(\boldsymbol{x}_{0})) \leq
\mathbb{E}_{q}\log 
\frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{0:T})}
$$
所以我们的目标变成优化负对数似然函数的上界：
$$
\min \quad \mathbb{E}_{q}\log 
\frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{0:T})}
$$
进一步化简，可以得到：
$$
\begin{align}

\mathbb{E}_{q}\log 
\frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{0:T})} 
&= 
\mathbb{E}_{q}\log
\frac{\prod_{t=1}^{T}q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})}
{p_{\theta}(\boldsymbol{x}_{T})\prod_{t=1}^{T}p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})}
\\ 
&= \mathbb{E}_{q}(
-\log p_{\theta}(\boldsymbol{x}_{T})+
\sum_{t=1}^{T}\log \frac{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})}{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})})

\end{align}
$$

继续化简括号里的第二项：
$$
\begin{align}

\sum_{t=1}^{T}\log \frac{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})}{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})}
&= \sum_{t=2}^{T}\log \frac{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})}{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})} + \log \frac{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}{p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})}
\\ &= \sum_{t=2}^{T}\log 
\frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{0})}
+ \log \frac{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}{p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})}
(markov+bayes)
\\ &= \sum_{t=2}^{T}\log 
\frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})}
+ \sum_{t=2}^{T}\log \frac{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{0})}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{0})}
+ \log \frac{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}{p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})}
\\ &= \sum_{t=2}^{T}\log 
\frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})}
+ \log \frac{q(\boldsymbol{x}_{T}|\boldsymbol{x}_{0})}{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}
+ \log \frac{q(\boldsymbol{x}_{1}|\boldsymbol{x}_{0})}{p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})}
\\ &= \sum_{t=2}^{T}\log 
\frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})}
+ \log \frac{q(\boldsymbol{x}_{T}|\boldsymbol{x}_{0})}{{p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})}}
\end{align}
$$
代入得到结果：
$$
\mathbb{E}_{q}\log 
\frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{0:T})} 
= \mathbb{E}_{q}(
\log \frac{q(\boldsymbol{x}_{T}|\boldsymbol{x}_{0})}{p_{\theta}(\boldsymbol{x}_{T})}
+ \sum_{t=2}^{T}\log 
\frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})}
{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})}
- \log p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})
)
$$
下面我们逐步来看这三项：

>$L_T$​​
>$$
>\log \frac{q(\boldsymbol{x}_{T}|\boldsymbol{x}_{0})}{p_{\theta}(\boldsymbol{x}_{T})} = KL(q(\boldsymbol{x}_{T}|\boldsymbol{x}_{0}) || p_{\theta}(\boldsymbol{x}_{T}))
>$$
>为q和p的KL散度，由于DDPM中方差为确定常量，所以该项KL散度为常量。

>$L_{t-1}$​​​​
>$$
>\sum_{t=2}^{T}\log 
>\frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t},\boldsymbol{x}_{0})}
>{p_{\theta}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t})} = 
>\sum KL(q||p_{\theta})
>$$
>该项为主要的优化目标，分子项为此前推导得到的后验概率，分母为逆向过程模型参与预测的分布。

>$L_0$​
>$$
>\log p_{\theta}(\boldsymbol{x}_{0}|\boldsymbol{x}_{1})
>$$
>直观上理解，这是最后一步逆向过程生成初始图片。
>
>该步本质上是将连续的高斯分布转换成离散的数值，同时$input.size = output.size$，每个位置表示的是对数概率，通过小范围缩放高斯分布CDF的值得到离散化的对数概率。

下面对第二项进行推导，首先对于对于两个高斯分布的KL散度
$$
KL(q||p) = \log \frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2\sigma_p^2}+ \frac12
$$
对于最小化$L_{t-1}$的目标，我们带入上式，有：
$$
\min \quad ||\tilde{\mu}_t(\boldsymbol{x}_{t}, \boldsymbol{x}_{0}) - \mu_{\theta}(\boldsymbol{x}_{t},t)||^2
$$
下面回到loss的选择上，我们选择对噪声进行预测，所以我们要讲优化转变成噪声的表示：
$$
\min \quad ||
\frac{1}{\sqrt{\alpha_t}}(
\boldsymbol{x}_{t} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon
) - 
\frac{1}{\sqrt{\alpha_t}}(
\boldsymbol{x}_{t} - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_{\theta}(\boldsymbol{x}_{t},t))
||^2
$$
化简后得到的优化问题可以被写成：
$$
\min \quad ||
\epsilon - \epsilon_{\theta}(\boldsymbol{x}_{t},t)
||^2
= ||
\epsilon - \epsilon_{\theta}(\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} + 
\sqrt{\bar{\alpha}_t}\boldsymbol{x}_{0},t)
||^2
$$



<img src="/Users/wangzirui/Diffusion4ECG/md-figure/DDPM-train--sample.png" alt="DDPM-train--sample" style="zoom:50%;" />



# Improved DDPM



## Code Arch



# Score-baesd model

`分数：`分数模型中使用梯度来表示分数，是对数概率密度函数的梯度方向。
$$
\frac{\partial \log p_{data}(x)}{\partial x}
$$
分数（方向）是符合物理直观的，一个数据样本的分布趋势是满足高概率密度函数梯度的。

**如果在采样过程中沿着分数的方向走，就能走到数据分布的高概率密度区域，最终生成的样本就会符合原数据分布。**

`朗之万动力学模型：`朗之万采样从一个已知的先验分布中采样一个初始样本，然后计算分数，将该样本向高分数（高概率密度函数）区域靠近。同时为了保证随机性（如果没有随机性，那生成的样本同质化严重，经过超多迭代都会收敛到最高对数概率密度函数最高的地方），朗之万采样方程带有随即项（布朗运动）。

------

我们从最后的目标出发，我们的优化目标为：
$$
\frac12 \mathbb{E}_{p_{data}} [||s_{\theta}(\boldsymbol{x}) - \nabla_{\boldsymbol{x}}\log p_{data}(\boldsymbol{x})||^2_2]
$$
其中，$\theta$表示模型参数，$s_{\theta}$表示预测网络，后一项梯度项为分数，$p_{data}$为真实分布。

对这样的优化问题显然存在一个问题就是我们未知真实分布，如果知道真实分布那也不用求了，所以我们先想办法去除掉真实分布项，我们采用分数匹配的方法得到的化简结果为：
$$
\mathbb{E}_{p_{data}}[tr(\frac{\partial s_{\theta}(\boldsymbol{x})}{\partial \boldsymbol{x}})
+ \frac12 ||s_{\theta}(\boldsymbol{x})||^2_2]
$$

> [!NOTE]
>
> 推导过程就是一个分部积分的过程，我们首先对优化目标的公式进行展开
> $$
> \frac12 \mathbb{E}_{p_{data}}[
> ||s_{\theta}(\boldsymbol{x})||^2_2 - 2(\frac{\partial \log p_{data}(\boldsymbol{x}) }{\partial \boldsymbol{x}})^T s_{\theta}(\boldsymbol{x}) + ||\frac{\partial \log p_{data}(\boldsymbol{x}) }{\partial \boldsymbol{x}}||^2_2
> ]
> $$
> 只有第一项和第二项带有$\theta$表示和网络有关，最后一项和模型参数无关，而第一项直接就可以通过模型计算出来，我们关注第二项，写成积分形式，我们有：（这里把$\boldsymbol{x}$看作向量，梯度项表示为标量对向量求导，得到的维度为$m \times 1$，而模型预测项和分数的维度大小一致，所以要有转置，最后相乘得到的结果维度为$1 \times 1$；在写法上，对于模型预测项得到的结果，多个维度可以看作是多个神经元输出的结果，所以结果写成如下形式）
> $$
> \int p_{data}(\boldsymbol{x})[-2(\frac{\partial \log p_{data}(\boldsymbol{x}) }{\partial \boldsymbol{x}})^T s_{\theta}(\boldsymbol{x})]d\boldsymbol{x} = 
> \int p_{data}(\boldsymbol{x})[\sum_{i} -2(\frac{\partial \log p_{data}(\boldsymbol{x}) }{\partial x_i})s_{\theta,i}(\boldsymbol{x})]d\boldsymbol{x}
> $$
> 继续化简，我们有：
> $$
> \sum_{i}\int p_{data}(\boldsymbol{x})[-2(\frac{\partial \log p_{data}(\boldsymbol{x}) }{\partial x_i})s_{\theta,i}(\boldsymbol{x})]d\boldsymbol{x} = 
> \sum_{i}\int -2p_{data}(\boldsymbol{x})\frac{1}{p_{data}(\boldsymbol{x})}
> \frac{\partial p_{data}(\boldsymbol{x})}{\partial x_i}s_{\theta,i}(\boldsymbol{x})d\boldsymbol{x}
> $$
> 所以，
> $$
> -2 \int \sum_i \frac{\partial p_{data}(\boldsymbol{x})}{\partial x_i}s_{\theta,i}(\boldsymbol{x})d\boldsymbol{x} = 
> -2 \int \sum_i (\frac{\partial (p_{data}(\boldsymbol{x}) s_{\theta,i}(\boldsymbol{x}))}{\partial x_i} - \frac{\partial s_{\theta,i}(\boldsymbol{x})}{\partial x_i}p_{data}(\boldsymbol{x}))d\boldsymbol{x}
> $$
> 我们作出假设$p(\infty) = 0$，
> $$
> -2 \int \sum_i (\frac{\partial (p_{data}(\boldsymbol{x}) s_{\theta,i}(\boldsymbol{x}))}{\partial x_i})d\boldsymbol{x} = p_{data}(\boldsymbol{x}) s_{\theta}(\boldsymbol{x})|^{+\infty}_{-\infty} = 0
> $$
> 继续化简，我们可以得到：
> $$
> -2 \int \sum_i (\frac{\partial (p_{data}(\boldsymbol{x}) s_{\theta,i}(\boldsymbol{x}))}{\partial x_i} - \frac{\partial s_{\theta,i}(\boldsymbol{x})}{\partial x_i}p_{data}(\boldsymbol{x}))d\boldsymbol{x} = 
> 2 \int \sum_i \frac{\partial s_{\theta,i}(\boldsymbol{x})}{\partial x_i}p_{data}(\boldsymbol{x})d\boldsymbol{x} \\
> = 2 \int p_{data}(\boldsymbol{x}) tr(\frac{\partial s_{\theta}(\boldsymbol{x})}{\partial \boldsymbol{x}})d\boldsymbol{x} 
> = 2\mathbb{E}_{p_{data}}(tr(\frac{\partial s_{\theta}(\boldsymbol{x})}{\partial \boldsymbol{x}}))
> $$
> 此时再加上最开始拆开分部积分的第一项喊模型参数的项，我们就可以得到最后的结果。

但是对于矩阵的trace来说，依旧存在较高复杂度，所以给出两种优化方案：

`Sliced Score Matching：`通过引入随机向量（均值为0，协方差为单位阵）来解决。

`Denosing Score Matching：`设定一个已知的分布去替代原始未知分布，用$q_\sigma(\widetilde{\boldsymbol{x}}) \approx p_{data}(\boldsymbol{x})$，其中：
$$
q_\sigma(\widetilde{\boldsymbol{x}}) = \int q_\sigma(\widetilde{\boldsymbol{x}}|\boldsymbol{x})p_{data}(\boldsymbol{x})d\boldsymbol{x}
$$
这要求我们用极小噪声处理数据，只有在噪声很小的情况下q分布和p分布才近似等同。所以我们的优化目标可以改写成：
$$
\frac12 \mathbb{E}_{q_{\sigma}}(||s_{\theta}(\widetilde{\boldsymbol{x}})
- \frac{\partial \log q_\sigma(\widetilde{\boldsymbol{x}})}{\partial \widetilde{\boldsymbol{x}}}||_2^2)
$$
已有论文证明：（ESM=DSM）
$$
\frac12 \mathbb{E}_{q_{\sigma}}(||s_{\theta}(\widetilde{\boldsymbol{x}})
- \frac{\partial \log q_\sigma(\widetilde{\boldsymbol{x}})}{\partial \widetilde{\boldsymbol{x}}}||_2^2)
= \frac12 \mathbb{E}_{q_{\sigma}(\widetilde{\boldsymbol{x}}|\boldsymbol{x})p_{data}(\boldsymbol{x})}(||s_{\theta}(\widetilde{\boldsymbol{x}})
- \frac{\partial \log q_\sigma(\widetilde{\boldsymbol{x}}|\boldsymbol{x})}{\partial \widetilde{\boldsymbol{x}}}||_2^2)
$$
其中我们预先设定的分布是$q_{\sigma}(\widetilde{\boldsymbol{x}}|\boldsymbol{x}) \sim \mathcal{N}(\widetilde{\boldsymbol{x}};\boldsymbol{x}, \sigma^2 \boldsymbol{I})$，写成直接的形式就是：
$$
\widetilde{\boldsymbol{x}} = \boldsymbol{x} + \sigma \boldsymbol{\epsilon},\quad 
\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})
$$
所以完全从分布的角度来看的情况下，喂给模型的均为加噪后的数据，模型学到的也是噪声分布，只有当噪声足够小的情况下，我们才能把学到的分布近似等同于真实分布。

模型通过分数训练完成后，我们通过朗之万采样采样获得样本，我们先看没有随机项的采样表示：
$$
\widetilde{\boldsymbol{x}}_{t} = \widetilde{\boldsymbol{x}}_{t-1} +
\frac{\partial \log q(\widetilde{\boldsymbol{x}}_{t-1})}{\partial \widetilde{\boldsymbol{x}}_{t-1}}
$$
其中$q$分布近似等同不加噪声的原始分布，t是取样的迭代步数，经过无穷步迭代，所有样本都会趋近于相同的位置，因为优化路线都是最大对数概率密度函数，所以我们需要加入随机项让采样得到的样本的多样性更强。
$$
\widetilde{\boldsymbol{x}}_{t} = \widetilde{\boldsymbol{x}}_{t-1} +
\frac{\epsilon}{2}s_{\theta}(\widetilde{\boldsymbol{x}}_{t-1})+
\sqrt{\epsilon}\boldsymbol{z}_{t-1}
$$
其中$\epsilon$是步长，$\boldsymbol{z}_{t}$是随机噪声，加入扰动后样本的多样性明显会更强，当步长趋近于0同时t趋近于无穷步数时，我们可以认为我们构造的样本基本等同于原始分布。

得到了训练策略和采样策略的大概思想，我们使用加噪的方案，下面我们给出具体的实现操作，首先对于loss。

> [!IMPORTANT]
>
> 在上文提到的加噪方案中，我们需要加一个**很小**的噪声来保证构造的新分布和原分布近似等同，但是对于分数匹配这一做法来说，很容易出现loss震荡，可以从**流形假设**来看：
>
> 流形假设说的是数据总是倾向分布于低维空间，简单来说就是数据的特征维度不是所有维度都是有用的，用一个二维平面直角坐标系表示一个圆心在原点的圆显然有一个维度多余了，如果用极坐标的话只需要一个参数就可以了。
>
> 对于分数匹配模型来说，分数（梯度）的计算是针对于全部的编码空间的，很有可能出现“没有用”的维度（对于分布而言，$p_{data}(\boldsymbol{x})$并没有被占满，即$\boldsymbol{x}$只占据了分布的一部分空间，形象来说，原始分布存在于整个二维平面直角坐标系，但是我们的输入$\boldsymbol{x}$只存在于直角坐标系中的一个单位矩形），在这种情况下可能就会出现loss震荡的情况。
>
> 一个合理的解决方案就是拓宽我们$\boldsymbol{x}$的分布范围，让$\boldsymbol{x}$尽可能分布在整个分布空间，这样的话就打破了流形。对于这样的想法可以使用高斯噪声，高斯噪声是分布在整个分布空间的，让$\boldsymbol{x}$加上高斯噪声就可以“打散”原来的$\boldsymbol{x}$的分布空间。
>
> 但是这样做的问题也存在，加噪声的前提是要加一个足够大的噪声，才能让$\boldsymbol{x}$分布在更大的原始分布空间。但这和原本的加噪方案加很小噪声有冲突，所以我们设计一个梯度噪声。

关于梯度噪声等级的设计，我们先用较大的噪声打散分布空间，然后用较小的噪声确保构造的分布和原始分布的相似性，所以我们的loss可以被写成：
$$
\frac12 \mathbb{E}_{q_{\sigma}(\widetilde{\boldsymbol{x}}|\boldsymbol{x})p_{data}(\boldsymbol{x})}(||s_{\theta}(\widetilde{\boldsymbol{x}})
- \frac{\partial \log q_\sigma(\widetilde{\boldsymbol{x}}|\boldsymbol{x})}{\partial \widetilde{\boldsymbol{x}}}||_2^2)
=
\frac12 \mathbb{E}_{q_{\sigma}(\widetilde{\boldsymbol{x}}|\boldsymbol{x})p_{data}(\boldsymbol{x})}(||s_{\theta}(\widetilde{\boldsymbol{x}},\sigma)
- \frac{\partial \log q_\sigma(\widetilde{\boldsymbol{x}}|\boldsymbol{x})}{\partial \widetilde{\boldsymbol{x}}}||_2^2)
$$
而我们预先设计的分布为$q_{\sigma}(\widetilde{\boldsymbol{x}}|\boldsymbol{x}) \sim \mathcal{N}(\widetilde{\boldsymbol{x}};\boldsymbol{x}, \sigma^2 \boldsymbol{I})$，所以我们有：
$$
f_{q}(\widetilde{\boldsymbol{x}}) = \frac{1}{\sqrt{2\pi}\sigma}e^{
-\frac{(\widetilde{\boldsymbol{x}} - \boldsymbol{x})^2}{2\sigma^2}
}
$$
进一步化简，
$$
\frac{\partial \log q_\sigma(\widetilde{\boldsymbol{x}}|\boldsymbol{x})}{\partial \widetilde{\boldsymbol{x}}}
=
\frac{
\partial (-\frac{(\widetilde{\boldsymbol{x}} - \boldsymbol{x})^2}{2\sigma^2})
}{\partial \widetilde{\boldsymbol{x}}}
=
\frac{\boldsymbol{x} - \widetilde{\boldsymbol{x}}}{\sigma^2}
$$
所以我们就得到某一个确定噪声等级下的loss，可以写成：
$$
\mathcal{L}(\theta, \sigma) = \frac12 \mathbb{E}_{q_{\sigma}(\widetilde{\boldsymbol{x}}|\boldsymbol{x})p_{data}(\boldsymbol{x})}(||s_{\theta}(\widetilde{\boldsymbol{x}},\sigma)
- \frac{\boldsymbol{x} - \widetilde{\boldsymbol{x}}}{\sigma^2}||_2^2)
$$
对于不同的噪声级别，我们进行加权组合，
$$
\mathcal{L}_{all} = \lambda_i 
$$










## SDE

一阶线性随机微分方程可以写为：
$$
dY = f(X, Y)dt
$$
随机微分方程的一般表达形式可以写为：（Ito process）
$$
dX(t) = f(X(t),t)dt + g(t)dW
$$
其中随机过程$\{X(t), t\in T=[0,T]\}$，$f,g$均为可定义的函数（在AI中为超参），$W(t)$为维纳过程/布朗运动，其中$W(t)\sim \mathcal{N}(0,c^2t)$，标准布朗运动的取值为$c=1$​。标准布朗运动的差分方程为：
$$
(W(t + \Delta t) - W(t) )\sim \mathcal{N}(0, \Delta t), \Delta t \rightarrow 0
$$
把Ito process写作更直观的diffusion方程：
$$
\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_{t} +
f_t(\boldsymbol{x}_{t})\Delta t + g_tdW
$$
根据差分方程对标准布朗运动的部分进行展开，我们有：
$$
\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_{t} +
f_t(\boldsymbol{x}_{t})\Delta t + g_t(\sqrt{\Delta t}\epsilon), \quad 
\epsilon \sim \mathcal{N}(0, \boldsymbol{I}), \quad
dW \sim \mathcal{N}(0, \Delta t)
$$
写成概率的形式，也就是采样的形式： 
$$
p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_{t}) = 
\mathcal{N}(\boldsymbol{x}_{t+\Delta t}; \boldsymbol{x}_{t} +
f_t(\boldsymbol{x}_{t})\Delta t, g_t^2\Delta t \boldsymbol{I})
$$


## DDPM和SMLD的统一性









\boldsymbol{x}_{t}







# TimeGrad

主要采用的是RNN的框架做时序预测，本质上就是reverse过程串行了一个RNN。



# CSDI



# SSSD









# VAE









# 生成模型总结







# Appendix

**信息熵**可以认为是信息量的期望：
$$
H_p = \sum_x p(x)\log \frac{1}{p(x)}
$$
其中$\log \frac{1}{p(x)}$可以被认为是信息量（概率越小，出现的可能越“惊喜”，信息量越大）

**交叉熵**衡量两个分布之间的差异：
$$
H(p,q) = \sum_x p(x)\log \frac{1}{q(x)}
$$
其中p是真实分布，q是预测分布。很多时候目标函数的目标就是最小化交叉熵，交叉熵的理论最小值就是两个分布完全等同的时候，这时候$H(p,q) = H_p$不为0。

**KL散度**也用来衡量两个分布之间的差异：
$$
KL(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p,q) - H_p
$$
我们用q分布去近似p分布时候需要多少额外信息量，这个信息量的值就是KL散度，当两个分布完全相同时，KL散度也为0.



**多元条件概率:**
$$
P(A,B,C) = P(C|A,B)P(A,B)=P(C|A,B)P(B|A)P(A)
$$

$$
P(B,C|A) = P(B|A)P(C|B,A)
$$

对于第二个公式，可以先考虑$P(B,C)$的分解。



**MLE和MAP**
$$
P(\theta|x) = \frac{P(x|\theta)P(\theta)}{P(x)}
$$
其中$\theta$是模型参数，$x$是数据，$P(\theta)$是先验概率，$P(\theta|x)$是后验概率，$P(x|\theta)$是似然函数。

对于MLE来说，目标任务是$\max P(x|\theta)$，其中$\theta$是变量，$x$为观测到的数据，常规做法是令$\log'(P(x|\theta)) = 0$解得$\theta$。

对于MAP来说，目标任务是$\max P(\theta|x)$，非常直观，给定确定的数据x下，求一个最大的$P(\theta|x)$。



**标量关于向量的导数**

若$\boldsymbol{x}\in \mathbb{R}^m, y \in \mathbb{R}$，（可以看作是$y = \boldsymbol{a}^T\boldsymbol{x}, \boldsymbol{a}\in \mathbb{R}^m$，一行乘一列）那么
$$
\frac{\partial y}{\partial \boldsymbol{x}} = [\frac{\partial y}{\partial x_1},
\frac{\partial y}{\partial x_2},\dots,\frac{\partial y}{\partial x_m}]^T
\in \mathbb{R}^{m\times 1}
$$
**向量关于标量的导数**

若$x\in \mathbb{R}, \boldsymbol{y}\in \mathbb{R}^n$，（可以看作是$\boldsymbol{y} = \boldsymbol{a}x, \boldsymbol{a}\in \mathbb{R}^n$）那么
$$
\frac{\partial \boldsymbol{y}}{\partial x} = [\frac{\partial y_1}{\partial x},
\frac{\partial y_2}{\partial x}, \dots, \frac{\partial y_n}{\partial x}]
\in \mathbb{R}^{1 \times n}
$$
**向量关于向量的导数**

若$\boldsymbol{x}\in \mathbb{R}^m, \boldsymbol{y}\in \mathbb{R}^n$，（可以看作是$\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}, \boldsymbol{A}\in \mathbb{R}^{n\times m}$）那么
$$
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = M_{Jacobian}^T =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \dots & 
\frac{\partial y_1}{\partial x_m}\\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \dots & 
\frac{\partial y_2}{\partial x_m}\\
\vdots & \vdots & \dots & \vdots \\
\frac{\partial y_n}{\partial x_1} & \frac{\partial y_n}{\partial x_2} & \dots & 
\frac{\partial y_n}{\partial x_m} & 
\end{bmatrix} ^T
\in \mathbb{R}^{m \times n}
$$


**常见向量导数**
$$
\frac{\partial \boldsymbol{a}^T\boldsymbol{X}\boldsymbol{b}}{\partial \boldsymbol{X}} = 
\boldsymbol{a}\boldsymbol{b}^T,\quad
\frac{\partial \boldsymbol{a}^T\boldsymbol{X}^T\boldsymbol{b}}{\partial \boldsymbol{X}} = 
\boldsymbol{b}\boldsymbol{a}^T
$$





