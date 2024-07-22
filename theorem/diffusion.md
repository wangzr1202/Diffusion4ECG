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
根据二次函数的展开式：
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

在此之前，我们需要先完成对MAP的优化，直接MAP不可求解，所以对于此类生成模型均为计算MLE，我们先构建一个似然函数。

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





# Improved DDPM



## Code Arch



# Score-baesd model

`梯度：`分数模型中使用梯度来表示分数，是对数概率密度函数的梯度方向。
$$
\frac{\partial \log p_{data}(x)}{\partial x}
$$
分数（方向）是符合物理直观的，一个数据样本的分布趋势是满足高概率密度函数梯度的。

**如果在采样过程中沿着分数的方向走，就能走到数据分布的高概率密度区域，最终生成的样本就会符合原数据分布。**



`朗之万动力学模型：`









# TimeGrad

主要采用的是RNN的框架做时序预测，本质上就是reverse过程串行了一个RNN。



# CSDI



# SSSD









# VAE









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

