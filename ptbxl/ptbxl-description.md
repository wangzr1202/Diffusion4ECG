行文逻辑

novelty：vq，augmentation，encoder，donwstream task

需不需要对比diffusion？encoder选择预测整个分布的模型？

encoder：cnn/rnn/

备选期刊：

nature machine intelligence --> npj digital medicine --> nature communications -->

TPAMI -->



当下任务

下游：对照指南，看ECG上能诊断出什么病，对比一下VQ的中间特征蕴含什么信息。

# Related work

## dataset description

该数据集包括来自**18885**名患者的**21837**份临床12导联心电图记录，长度为10秒。该数据集在性别方面是平衡的(52%的男性和48%的女性)，涵盖了从0到95岁的整个年龄范围(中位数62)。

心电图记录由最多两名心脏病专家进行注释，在符合SCP-ECG标准的**71**种不同的心电图statement中，有多种心电图statement。这些statement以统一的、机器可读的形式涵盖了形式、节奏和诊断。对于诊断标签，提供了5个粗超类和24个子类的分层组织。

ECG statements（心电图声明）是指在心电图数据中记录的关于心脏状态的描述或注释。这些声明是由医疗专家基于心电图波形的特征进行的判读和诊断，反映了心脏的电活动和可能存在的心脏病理状态。具体来说，ECG statements 通常包括以下几个方面的内容：

1.  **诊断声明（Diagnostic Statements）**：

    *   描述心电图中反映的具体心脏疾病或异常。例如，可能会标注出心肌梗死、心房颤动、左心室肥大等具体的心脏病诊断。这些诊断通常按照某种标准（如SCP-ECG标准）进行分类，以确保数据的结构化和一致性。

2.  **形态声明（Form Statements）**：

    *   这些声明描述心电图波形的形态学特征，例如P波、QRS波群、T波的形状、持续时间和幅度。形态声明用于识别和描述心电图中各个波形的形态特征，这些特征对于确定心脏的电活动和结构异常至关重要。

3.  **节律声明（Rhythm Statements）**：

    *   节律声明关注的是心脏的电活动的时间和规律性，例如是否存在窦性心律、房性或室性心律失常等。这些声明帮助判断心跳的节奏是否正常或是否存在心律失常。

这些ECG statements 是心电图分析的重要组成部分，可以帮助医生识别和诊断潜在的心脏问题，并且在数据集中，这些声明被用于训练和评估机器学习模型，以便自动化地分析心电图数据。

数据集直接分了10个folder，10个folder中的不同类别相对均衡。

（来自ptbxl原文）

## model

对比的baseline

### baseline

文章说明从cv领域迁移到ECG领域的框架并能很好的适配，复杂框架相比较**ResNet-18**并没有本质的提升。

（In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis）

**Bi-LSTM**

**transformer**

\*\*SSM: \*\*

### Diffusion

某些任务上的比对对象，diffusion过程大都用来重建，默认为信号重建任务，但是也可以用来做条件生成等任务。

[Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/pdf/2107.03006): 对离散状态建模的diffusion，通过one-hot+状态转移矩阵建模，可以对vq工作的离散空间进行建模。

SSSD（S4 for encoder），信号重建效果好，S4模块工作效率较高.

（Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models）

CSDI（）

### vqvae

待解决的问题：1.调研ECG中vqvae的使用情况☑️；2.调研相似信号（语音等）中vqvae的使用情况；3.选择一个合适的vqvae框架；4.endoer和decoder怎么选；5.decoder怎么重建波形（分阶段比如先粗后细还是直接一步生成，自回归是否对长序列生成有额外的优势？）

**quantization on ECG:** 大部分是量化减小模型体量，方便部署到边缘设备和方便计算。

related work可以引用2-3篇，谷歌学术检索词“quantization AND ECG”

***

\*\*vqvae on ECG: \*\*只有一篇韩国的文章，但是主要面对的还是可解释性，提出新的指标来衡量ROC曲线。

[CAN WE EXPLAIN IT: DETERMINING TIME SERIES DATA INTERPRETABILITY WITH CLASSIFICATION OF NEURAL DISCRETE REPRESENTATION](https://xai.kaist.ac.kr/static/files/2023_xai_workshop/paper25.pdf): 使用vqvae

***

**vqvae on speech waveform, vqvae on audio, vqvae on voice**

\*\*vqvae on video, audio: \*\*

videoGPT (2D -- 2D)，实现的任务是图像生成和视频生成，两阶段，第一阶段为原始的vqvae过程（视频输入，编码隐空间，解码重构输入），第二阶段输入从隐空间采样，使用transformer自回归预测视频。

（VideoGPT: Video Generation using VQ-VAE and Transformers）

specVQGAN (2D -- 1D)，通过图片生成1D语音，transformer编码输入特征，然后离散codebook表征，然后解码生成频谱图，最后生成语音。

（[Taming Visually Guided Sound Generation](https://arxiv.org/pdf/2110.08791)）

\*Enhancing Codec LN (1D -- 1D)，语音重建任务，有噪声条件，整体框架就是一个vqvae。对输入的多路speech做简单平均得到一个全新的speaker signal，计算vq时候同时考虑speech/speaker。

（ENHANCING INTO THE CODEC: NOISE ROBUST SPEECH CODING WITH VECTOR-QUANTIZED AUTOENCODERS）

\*\*vqvae on image: \*\*

vqvae：初始论文大量工作就是在图像上，也做了语音和视频的实验

（Neural Discrete Representation Learning）

vqGAN：

### signal processing

### others

\*\*trick: \*\*

Data augmentation：

使用增强的时候可以考虑多导联本身的情况，因为增强导联（avr）本质上也没有提供新的信息，就是一个额外视角。

\*[A Systematic Survey of Data Augmentation of ECG Signals for AI Applications](https://www.mdpi.com/1424-8220/23/11/5237): 调查常用的ecg数据集和常见的数据增强方案。

[A novel data augmentation approach for enhancement of ECG signal classification](https://www.sciencedirect.com/science/article/pii/S1746809423005475): 把ECG切割成片段然后重新排列作为数据增强，但是这篇文章是将信号做成图片然后输入给CNN做分类，作者也提出一个新的计算成本较低的四层CNN。

[The Effect of Data Augmentation on Classification of Atrial Fibrillation in Short Single-Lead ECG Signals Using Deep Neural Networks](https://arxiv.org/pdf/2002.02870): 使用GMM和GAN做数据增强，主要目的是解决类别不平衡，任务是AF（心房颤动）分类，发现有提升。

[Self-Supervised Learning with Attention-based Latent Signal Augmentation for Sleep Staging with Limited Labeled Data](https://www.ijcai.org/proceedings/2022/0537.pdf)

并行结构

codebook改进

训练loss改进

waveNet（2016deepmind自回归音频生成），waveRNN（），

## downstream task

**分类：**

[MSGformer: A multi-scale grid transformer network for 12-lead ECG arrhythmia detection](https://www.sciencedirect.com/science/article/abs/pii/S1746809423009321): 对肢体导联和胸导联分别进行特征提取，然后使用transormer进行分类（没细看）。

**信号重建：**

*   **信号生成**：这是一个与 NLP 中文本生成类似的任务。使用 VQ-VAE 提供的量化索引，通过自回归模型（如 Transformer 或 GPT）生成新的 ECG 信号片段。可以生成特定条件下的 ECG 数据，模拟不同病理或心律状态下的 ECG 表现。

*   可以考虑一些图片辅助生成语音的文章，可以考虑一些**聚类**方法

*   预测未来会不会发生心律失常等疾病

**判断两个ecg是不是属于一个人：**

(ElectroCardioGuard: Preventing Patient Misidentification in Electrocardiogram Databases through Neural Networks)

**分割与边界检测**

*   **P、QRS、T 波分割**：类似于计算机视觉中的图像分割任务。可以使用量化特征来进行 P 波、QRS 复合波和 T 波的自动分割。这在心电分析中非常重要，尤其是对于时间域的特征提取。

*

**异常检测**

*   **心电异常检测**：使用 VQ-VAE 提取的低维特征进行异常检测，类似于 NLP 中的语义异常检测任务。可以通过无监督或半监督方法，检测出与正常心电波形模式不同的异常数据（如异常心律、心肌缺血等）。

**迁移学习**

*   **跨任务迁移学习**：与计算机视觉领域中的图像特征迁移类似，VQ-VAE 提取的心电信号特征可以用作其他任务（如不同病理条件下的分类或检测任务）的初始特征。通过迁移学习，可以减少对标注数据的需求。

*   **心电特征嵌入表示**：类似于 NLP 中的词向量嵌入（如 Word2Vec）。可以通过量化后的特征生成心电信号的低维表示，并用于其他下游任务，如心电信号相似度计算、心电信号聚类分析等。

**信号压缩与恢复**

*   **心电信号压缩**：使用 VQ-VAE 提取的量化索引可以用于心电信号的高效压缩和存储，类似于图像或视频压缩技术。这可以在长时间监测心电信号时，减少数据的存储和传输开销。

*   **信号重构与去噪**：与计算机视觉中的图像去噪类似，可以使用量化的特征进行心电信号的重构和去噪，消除传感器噪声或其他干扰信号。

**序列对比学习**

*   **无监督对比学习**：类似于计算机视觉中的 SimCLR 或 NLP 中的对比学习任务。通过对比心电信号的不同时间段或不同导联信号的相似性，训练模型学习更具区分度的表示。

**多模态融合**

*   **跨模态数据融合**：将心电信号的量化索引与其他生物信号（如血压、血氧、脑电图等）进行融合，进行多模态分析，类似于计算机视觉中的图文融合或 NLP 中的多模态任务。这有助于提供更加全面的生理学评估。

**聚类任务**

*   **患者心电聚类**：基于 VQ-VAE 生成的特征，可以将患者心电信号进行无监督聚类，识别出心电信号模式相似的患者群体，类似于图像聚类任务。这可以用于个性化的病理分析或群体健康风险评估。

*   **分类任务**：心律失常检测、病理分类

*   **生成任务**：心电信号生成、心律演化预测

*   **分割任务**：P、QRS、T 波分割

*   **异常检测**：心电异常检测

*   **时间序列预测**：心电信号的短期或长期预测

*   **无监督学习**：信号聚类、对比学习、异常检测

*   **多模态分析**：结合其他生理信号，做更广泛的生物数据分析。
