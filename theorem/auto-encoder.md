# VAE

<img src="/Users/wangzirui/Diffusion4ECG/md-figure/VAE-process.png" alt="VAE-process" style="zoom:15%;" />

假设我们有一系列随机变量$\boldsymbol{x}_{i}$，我们借用隐变量$\boldsymbol{z}$来完成生成任务，首先要通过输入的随机变量生成隐变量，我们设计一个新的分布$q_{\phi}$来近似$p_{\theta}$（直接求解通过贝叶斯公式，但是贝叶斯公式中表达出来的$p(\boldsymbol{x})$为一个复杂的多重积分不可求）


$$
p_{\theta}(\boldsymbol{z}|\boldsymbol{x}) \approx q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
$$

所以我们的优化目标就变成两个分布之间的差异：
$$
Object = KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||
p_{\theta}(\boldsymbol{z}|\boldsymbol{x}))
=-\sum_{\boldsymbol{z}}q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
\log \frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}
$$
我们有：
$$
\begin{align}
-\sum_{\boldsymbol{z}}q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
\log \frac{p_{\theta}(\boldsymbol{z}|\boldsymbol{x})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}
&=
-\sum_{\boldsymbol{z}}q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
(
\log \frac{p_{\theta}(\boldsymbol{z},\boldsymbol{x})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} - 
\log p_{\theta}(\boldsymbol{x})
)\\
&=
KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p_{\theta}(\boldsymbol{z},\boldsymbol{x}))
+ \log p_{\theta}(\boldsymbol{x})\\
\end{align}
$$
可以写成：
$$
KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p_{\theta}(\boldsymbol{z}|\boldsymbol{x}))
= KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p_{\theta}(\boldsymbol{z},\boldsymbol{x}))
+ \log p_{\theta}(\boldsymbol{x})
$$
所以优化目标就变成了：
$$
\mathcal{max} \quad \mathcal{L}(\theta,\phi;\boldsymbol{x}) = -KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p_{\theta}(\boldsymbol{z},\boldsymbol{x}))
$$
进一步化简：
$$
\begin{align}
\mathcal{L}(\theta,\phi;\boldsymbol{x})
&=
\sum_{\boldsymbol{z}}q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
\log \frac{p_{\theta}(\boldsymbol{z},\boldsymbol{x})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}\\
&=
\sum_{\boldsymbol{z}}q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
\log \frac{p_{\theta}(\boldsymbol{x}|\boldsymbol{z})p_{\theta}(\boldsymbol{z})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}\\
& =
\sum_{\boldsymbol{z}}q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
(\log \frac{p_{\theta}(\boldsymbol{z})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}
+ \log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})
)\\
& = 
-KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p_{\theta}(\boldsymbol{z}))
+ \underbrace{\mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} \log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})}_{Reconstruction-Loss}
\end{align}
$$

> [!NOTE]
>
> 对于上式倒数第二个等式(9)，我们
>
> a. **假定括号内的变量与$\phi$无关**，那么我们计作：
> $$
> \mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[f(\boldsymbol{z})]
> $$
> 在bp计算梯度（对参数求梯度）时，
> $$
> \begin{align}
> \nabla_{\phi}\mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[f(\boldsymbol{z})]
> &= \nabla_{\phi} \int q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) f(\boldsymbol{z}) d\boldsymbol{z}\\
> &= \int \nabla_{\phi} q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) f(\boldsymbol{z}) d\boldsymbol{z}\\
> &= \int q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) \nabla_{\phi} \log q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) f(\boldsymbol{z}) d\boldsymbol{z}\\
> &= \mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\nabla_{\phi} \log q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) f(\boldsymbol{z})]
> \end{align}
> $$
>
> 然后我们通过蒙特卡洛采样求解梯度：
> $$
> \frac1L \sum_{l=1}^{L}(\nabla_{\phi} \log q_{\phi}(\boldsymbol{z}^{(l)}|\boldsymbol{x}) f(\boldsymbol{z}^{(l)}))
> $$
> b. 若括号内的变量**与$\phi$有关**，我们只能写成：
> $$
> \mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[f_{\phi}(\boldsymbol{z})]
> $$
> 所以，
> $$
> \begin{align}
> \nabla_{\phi}\mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[f_{\phi}(\boldsymbol{z})]
> &= \int \nabla_{\phi} q_{\phi}(\boldsymbol{z}|\boldsymbol{x}) f_{\phi}(\boldsymbol{z}) d\boldsymbol{z}\\
> &= \int (f_{\phi}(\boldsymbol{z}) \nabla_{\phi} q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
> + q_{\phi}(\boldsymbol{z}|\boldsymbol{x})\nabla_{\phi} f_{\phi}(\boldsymbol{z})
> )d\boldsymbol{z}\\
> &= \int f_{\phi}(\boldsymbol{z}) \nabla_{\phi} q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
> d\boldsymbol{z}
> + \mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\nabla_{\phi} f_{\phi}(\boldsymbol{z})]\\
> &= \underbrace{\mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[f_{\phi}(\boldsymbol{z})\nabla_{\phi} \log q_{\phi}(\boldsymbol{z}|\boldsymbol{x})]}_{\mathcal{part1}}
> + \underbrace{\mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\nabla_{\phi} f_{\phi}(\boldsymbol{z})]}_{\mathcal{part2}}
> \end{align}
> $$
>
> - 对于$\mathcal{part1}$来说，和此前的假设的求解结果一致，可以通过蒙特卡洛求解，但是求解的过程中，由于log项的存在，在计算梯度时的数值会较小，log的值在0处较小会导致方差较大，所以该项需要被优化。
> - 对于$\mathcal{part2}$来说，直接通过蒙特卡洛求解梯度即可。
>
> 使用***reparameterization trick***对$\mathcal{part1}$进行优化，根本目的是希望梯度计算更稳定，方差更小。
>
> 我们可以使用一个新的随机变量解耦，（这里假定为高斯分布）
> $$
> \boldsymbol{z} = \boldsymbol{\sigma} \boldsymbol{\epsilon} + \boldsymbol{\mu}
> , \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})
> $$
> 这样的话所有的随机性都转移到$\boldsymbol{\epsilon}$上，对$\boldsymbol{z}$的梯度计算更容易。同时，我们用新的随机变量替代$\boldsymbol{z}$。
> $$
> \boldsymbol{z} = g_{\phi}(\boldsymbol{\epsilon}, \boldsymbol{x})
> $$
>
> $$
> \mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[f_{\phi}(\boldsymbol{z})]
>  = \mathbb{E}_{q_{\phi}(\boldsymbol{\epsilon})}[f_{\phi}(g_{\phi}(\boldsymbol{\epsilon}, \boldsymbol{x}))]
> $$
>
> 下面我们再计算梯度：
> $$
> \nabla_{\phi}\mathbb{E}_{q_{\phi}(\boldsymbol{\epsilon})}[f_{\phi}(g_{\phi}(\boldsymbol{\epsilon}, \boldsymbol{x}))]
> = \mathbb{E}_{q_{\phi}(\boldsymbol{\epsilon})}[\nabla_{\phi}f_{\phi}(g_{\phi}(\boldsymbol{\epsilon}, \boldsymbol{x}))]
> $$
> 利用蒙特卡洛求解：
> $$
> \mathbb{E}_{q_{\phi}(\boldsymbol{\epsilon})}[\nabla_{\phi}f_{\phi}(g_{\phi}(\boldsymbol{\epsilon}, \boldsymbol{x}))] 
> = \frac1L \sum_{l = 1}^{L} \nabla_{\phi}f_{\phi}(g_{\phi}(\boldsymbol{\epsilon}^{(l)}, \boldsymbol{x}))
> $$
> 

<img src="/Users/wangzirui/Diffusion4ECG/md-figure/VAE-bp.png" alt="VAE-bp" style="zoom:45%;" />



所以对于我们的优化目标$\mathcal{L}(\theta,\phi;\boldsymbol{x})$，我们有：
$$
\begin{align}
\mathcal{L}(\theta,\phi;\boldsymbol{x})
&=
-KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p_{\theta}(\boldsymbol{z}))
+ \mathbb{E}_{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})} \log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})\\
\nabla_{\theta, \phi} \mathcal{L}(\theta,\phi;\boldsymbol{x})&= 
\underbrace{-\nabla_{\theta, \phi} KL(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p_{\theta}(\boldsymbol{z}))}_{Analytically \, Compute} 
+ 
\underbrace{\nabla_{\theta, \phi}\frac1L \sum_{l=1}^{L}\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z}^{(l)})}_{Monte\,Carlo\,Sample}
\end{align}
$$
前一项是可以直接给出解析解的，因为我们对于两个分布均假设为高斯分布，第一个分布为$\mathcal{N}(\mu, \sigma)$，第二个分布为$\mathcal{N}(0, I)$，我们很容易得到两个高斯分布之间的KL散度的梯度。

后一项为重构loss，计算重构量和输入量的MSE。

VAE的整体工作流程为：
$$
\begin{align*}
\boldsymbol{\mu}_{x}, \boldsymbol{\sigma}_{x} &= \mathcal{encoder}(\boldsymbol{x})\\
\boldsymbol{\epsilon} &\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\\
\boldsymbol{z} &= \boldsymbol{\sigma}_{x} \boldsymbol{\epsilon} + \boldsymbol{\mu}_{x}\\
\boldsymbol{x}_{r} &= \mathcal{decoder}(\boldsymbol{z})\\
\mathcal{loss}_{recons.} &= \mathbb{MSE}(\boldsymbol{x}_{r}, \boldsymbol{x})\\
\mathcal{loss}_{var.} &= -KL(\mathcal{N}(\boldsymbol{\mu}_{x}, \boldsymbol{\sigma}_{x})||\mathcal{N}(\boldsymbol{0}, \boldsymbol{I}))\\
\mathcal{L} &= \mathcal{loss}_{recons.} + \mathcal{loss}_{var.}\\
\end{align*}
$$




# VQ-VAE



​				   	 **codebook (8192*N)**

​						   	**|**

X --> **Encoder** --> Feature Map -(align)-> mode Feature Map --> **Decoder** --> X' -(loss)-> reconstruct X



<img src="/Users/wangzirui/Diffusion4ECG/md-figure/VQ-VAE.png" alt="VQ-VAE" style="zoom:40%;" />





# VQVAE-2









# VQGAN





