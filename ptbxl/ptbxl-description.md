行文逻辑

novelty：vq，augmentation，encoder，donwstream task

需不需要对比diffusion？encoder选择预测整个分布的模型？

encoder：cnn/rnn/





# Related work



## dataset description

该数据集包括来自**18885**名患者的**21837**份临床12导联心电图记录，长度为10秒。该数据集在性别方面是平衡的(52%的男性和48%的女性)，涵盖了从0到95岁的整个年龄范围(中位数62)。

心电图记录由最多两名心脏病专家进行注释，在符合SCP-ECG标准的**71**种不同的心电图statement中，有多种心电图statement。这些statement以统一的、机器可读的形式涵盖了形式、节奏和诊断。对于诊断标签，提供了5个粗超类和24个子类的分层组织。

ECG statements（心电图声明）是指在心电图数据中记录的关于心脏状态的描述或注释。这些声明是由医疗专家基于心电图波形的特征进行的判读和诊断，反映了心脏的电活动和可能存在的心脏病理状态。具体来说，ECG statements 通常包括以下几个方面的内容：

1. **诊断声明（Diagnostic Statements）**：
   - 描述心电图中反映的具体心脏疾病或异常。例如，可能会标注出心肌梗死、心房颤动、左心室肥大等具体的心脏病诊断。这些诊断通常按照某种标准（如SCP-ECG标准）进行分类，以确保数据的结构化和一致性。

2. **形态声明（Form Statements）**：
   - 这些声明描述心电图波形的形态学特征，例如P波、QRS波群、T波的形状、持续时间和幅度。形态声明用于识别和描述心电图中各个波形的形态特征，这些特征对于确定心脏的电活动和结构异常至关重要。

3. **节律声明（Rhythm Statements）**：
   - 节律声明关注的是心脏的电活动的时间和规律性，例如是否存在窦性心律、房性或室性心律失常等。这些声明帮助判断心跳的节奏是否正常或是否存在心律失常。

这些ECG statements 是心电图分析的重要组成部分，可以帮助医生识别和诊断潜在的心脏问题，并且在数据集中，这些声明被用于训练和评估机器学习模型，以便自动化地分析心电图数据。

数据集直接分了10个folder，10个folder中的不同类别相对均衡。

（来自ptbxl原文）



## task and metric



### metric

**Hamming Loss**

衡量预测标签和真实标签之间的差异。它计算模型在单个标签上的错误分类率（包括假阳性和假阴性），并将其平均到所有标签上。
$$
\text{Hamming Loss} = \frac{1}{N \times L} \sum_{i=1}^{N} \sum_{j=1}^{L} \mathbb{1}\{y_{ij} \neq \hat{y}{ij}\}
$$
其中，N 是样本数，L 是标签数，$y{ij} $和 $\hat{y}_{ij} $分别是真实标签和预测标签。范围从 0 到 1，值越小表示模型性能越好。

**Exact Match Ratio (Accuracy)**

也称为子集准确率，是指模型预测结果与真实标签完全匹配的比例。
$$
\text{Exact Match Ratio} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\{Y_i = \hat{Y}_i\}
$$
其中，$Y_i $和 $\hat{Y}_i$分别是真实标签集和预测标签集。范围: 0 到 1，值越高表示模型性能越好。注意: 该指标非常严格，因为只要有一个标签预测错误，该样本即被视为错误分类。

**precision**

Micro Precision在全局范围内计算精确率，不考虑单个标签，而是将所有样本和标签作为一个整体来计算。

首先在所有标签和所有样本的层面上累加True Positives（真正类，TP）和False Positives（假正类，FP）的总数，然后根据累加的总数计算精确率。
$$
\text{Micro Precision} = \frac{\sum_{i=1}^{L} \text{True Positives}_i}{\sum_{i=1}^{L} (\text{True Positives}_i + \text{False Positives}_i)}
$$
Micro Precision能够更好地处理样本中标签不平衡的问题，因为它对所有标签的预测错误同等重视。

在关注整体预测性能时，Micro Precision是一种很好的度量方法。

**Macro Precision**首先为每个标签单独计算精确率，然后取这些精确率的平均值，不考虑每个标签的样本数量。对每个标签单独计算精确率，然后对所有标签的精确率取平均值。
$$
\text{Macro Precision} = \frac{1}{L} \sum_{i=1}^{L} \frac{\text{True Positives}_i}{\text{True Positives}_i + \text{False Positives}_i}
$$

Macro Precision对所有标签一视同仁，无论标签在数据集中出现的频率如何。它更好地反映了模型在所有类别上的整体表现。特别适合处理多标签分类中的标签不平衡问题，因为它不会被高频标签所主导。由于对每个标签的贡献是平均的，Macro Precision可能会被表现较差的标签拉低，从而不完全反映高频标签的分类性能。





1. **[The ECG Signal Monitoring System Using Machine Learning Methods and LoRa Technology](https://www.par.pl/en/content/download/15455/191436/file/ECGSignalMonitoringSystemUsingMachineLearningMethodsLoRaTechnology.pdf)**  
   *Published:* 2024  
   *Summary:* This paper focuses on the classification of ECG signals using the PTB-XL database, employing machine learning methods integrated with LoRa technology.

2. **[Design of an Integrated Arrhythmia Detection Model using Connectivity Features and Multivariate Time Series Classification](https://wseas.com/journals/isa/2024/a185109-004(2024).pdf)**  
   *Published:* 2024  
   *Summary:* This research uses the PTB-XL dataset and a multivariate time series classification approach to detect arrhythmia.

3. **[Design of a novel recurrent neural network design with augmented activation for classification of ECG signals](https://www.academia.edu/download/89058563/Design_of_a_novel_recurrent_neural_network_design_with_augmented_activation_for_classification_of_ECG_signals.pdf)**  
   *Published:* 2024  
   *Summary:* This study introduces a novel recurrent neural network model for ECG signal classification, validated on multiple datasets including PTB-XL.

4. **[Interpretable Hybrid Multichannel Deep Learning Model for Heart Disease Classification Using 12-leads ECG Signal](https://ieeexplore.ieee.org/abstract/document/10579752/)**  
   *Published:* 2024  
   *Summary:* The paper discusses a deep learning model that improves heart disease classification using the PTB-XL dataset.

5. **[Improving deep-learning electrocardiogram classification with an effective coloring method](https://www.sciencedirect.com/science/article/pii/S0933365724000514)**  
   *Published:* 2024  
   *Summary:* The research enhances ECG classification using the PTB-XL dataset by applying a new coloring method.







## model

对比的baseline

### CNN

文章说明从cv领域迁移到ECG领域的框架并能很好的适配，复杂框架相比较**ResNet-18**并没有本质的提升。

（In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis）





### RNN

Bi-LSTM



**RNN with attention: **





**SSM: **



### Diffusion

某些任务上的比对对象，diffusion过程大都用来重建，默认为信号重建任务。

SSSD（S4 for encoder），信号重建效果好，S4模块工作效率较高

（Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models）



CSDI（）







### vqvae

待解决的问题：1.调研ECG中vqvae的使用情况☑️；2.调研相似信号（语音等）中vqvae的使用情况；3.选择一个合适的vqvae框架；4.endoer和decoder怎么选；5.decoder怎么重建波形（分阶段比如先粗后细还是直接一步生成，自回归是否对长序列生成有额外的优势？）



**quantization on ECG:** 大部分是量化减小模型体量，方便部署到边缘设备和方便计算。

related work可以引用2-3篇，谷歌学术检索词“quantization AND ECG”



------

**vqvae on ECG: **只有一篇韩国的文章，但是主要面对的还是可解释性，提出新的指标来衡量ROC曲线。

（CAN WE EXPLAIN IT: DETERMINING TIME SERIES DATA INTERPRETABILITY WITH CLASSIFICATION OF NEURAL DISCRETE REPRESENTATION）

------

**vqvae on speech waveform, vqvae on audio, vqvae on voice**



**vqvae on video, audio: **

videoGPT (2D -- 2D)，实现的任务是图像生成和视频生成，两阶段，第一阶段为原始的vqvae过程（视频输入，编码隐空间，解码重构输入），第二阶段输入从隐空间采样，使用transformer自回归预测视频。

（VideoGPT: Video Generation using VQ-VAE and Transformers）

specVQGAN (2D -- 1D)，通过图片生成1D语音，transformer编码输入特征，然后离散codebook表征，然后解码生成频谱图，最后生成语音。

（Taming Visually Guided Sound Generation）

*Enhancing Codec LN (1D -- 1D)，语音重建任务，有噪声条件，整体框架就是一个vqvae。对输入的多路speech做简单平均得到一个全新的speaker signal，计算vq时候同时考虑speech/speaker。

（ENHANCING INTO THE CODEC: NOISE ROBUST SPEECH CODING WITH VECTOR-QUANTIZED AUTOENCODERS）





**vqvae on image: **

vqvae：初始论文大量工作就是在图像上，也做了语音和视频的实验

（Neural Discrete Representation Learning）

vqGAN：





### ECG compression







**trick: **

Data augmentation：mask

并行结构

codebook改进

训练loss改进



### others

waveNet（2016deepmind自回归音频生成），waveRNN（），



##  downstream task

**分类：**





**信号重建：**





**判断两个ecg是不是属于一个人：**

(ElectroCardioGuard: Preventing Patient Misidentification in Electrocardiogram Databases through Neural Networks)
