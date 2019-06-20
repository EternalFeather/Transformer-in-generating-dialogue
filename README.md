An Implementation of Attention is all you need with Chinese Corpus
===
&emsp;&emsp;The code is an implementation of Paper [Attention is all you need](https://arxiv.org/abs/1706.03762) working for dialogue generation tasks like: **Chatbot**、 **Text Generation** and so on.  
&emsp;&emsp;**Thanks to every friends who have raised issues and helped solve them. Your contribution is very important for the improvement of this project. Due to the limited support of the 'static graph mode' in coding, we decided to move the features to 2.0.0-beta1 version. However if you worry about the problems from docker building and service creation with version issues, we still keep an old version of the code written by eager mode using tensorflow 1.12.x version to refer.**

# Documents
```
|-- root/
    |-- data/
        |-- src-train.csv
        |-- src-val.csv
        |-- tgt-train.csv
        `-- tgt-val.csv
    |-- old_version/
        |-- data_loader.py
        |-- eval.py
        |-- make_dic.py
        |-- modules.py
        |-- params.py
        |-- requirements.txt
        `-- train.py
    |-- tf1.12.0-eager/
        |-- bleu.py
        |-- main.ipynb
        |-- modules.py
        |-- params.py
        |-- requirements.txt
        `-- utils.py
    |-- images/
    |-- bleu.py
    |-- main-v2.ipynb
    |-- modules-v2.py
    |-- params.py
    |-- requirements.txt
    `-- utils-v2.py
```

# Requirements
- Numpy >= 1.13.1
- Tensorflow-gpu == 1.12.0
- **Tensorflow-gpu == 2.0.0-beta1**
    - cudatoolkit >= 10.0
    - cudnn >= 7.4 
    - nvidia cuda driver version >= 410.x
- tqdm
- nltk
- jupyter notebook

# Construction
&emsp;&emsp;As we all know the Translation System can be used in implementing conversational model just by replacing the paris of two different sentences to questions and answers. After all, the basic conversation model named "Sequence-to-Sequence" is develped from translation system. Therefore, why we not to improve the efficiency of conversation model in generating dialogues?  

<div align='center'>
    <img class="course-image" src="https://github.com/EternalFeather/Transformer-in-generating-dialogue/blob/master/images/transformer.png">
</div>

&emsp;&emsp;With the development of [BERT-based models](https://arxiv.org/abs/1810.04805), more and more nlp tasks are refreshed constantly. However, the language model is not contained in BERT's open source tasks. There is no doubt that on this way we still have a long way to go.  

## Model Advantages
&emsp;&emsp;A transformer model handles variable-sized input using stacks of self-attention layers instead of RNNs or CNNs. This general architecture has a number of advantages and special ticks. Now let's take them out:

- It make no assumptions about the temporal/spatial relationships across the data.(However this was proved to be not sure from AutoML)
- Layer outputs can be calculated in parallel, instead of a series like an RNN.(Faster training)
- Distant items can affect each other's output without passing through many RNN-steps, or CNN layers.(Lower cost)
- It can learn long-range dependencies, which is a challenge of dialogue system.

## Implementation details
&emsp;&emsp;**In the newest version of our code**, we complete the details described in paper.  

### Data Generation
- Use tfrecord to unified data storage format.
- Use dataset to load the processed chinese token datasets.

### Positional Encoding
- Since the model doesn't contain any memory mechanism, positional encoding is added to give it some information about the relative position of the words in the sentence by representing a token into a d-dimensional space where tokens with similar meaning will be closer to each other.  

<div align='center'>
    <img src="http://latex.codecogs.com/gif.latex?\Large{PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})}" />
</div>

<div align='center'>    
    <img src="http://latex.codecogs.com/gif.latex?\Large{PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})}" />
</div>

### Mask
- We create two different type of mask during training. One is for the padding masking, the other is for the decoder look_ahead masking to keep the following tokens invisible when generating the previous ones.

### Scaled dot product attention
- The attention function used by the transformer takes three inputs: Q,K,V. The equation used to calculate the attention weights, which is scaled by a factor of square root of the depth is:  

<div align='center'>
    <img src="http://latex.codecogs.com/gif.latex?\Large{Attention(Q, K, V) = softmax_k(\frac{QK^T}{\sqrt{d_k}}) V}" />  
</div>

<div align='center'>
    <img class="course-image" src="https://github.com/EternalFeather/Transformer-in-generating-dialogue/blob/master/images/scaled_attention.png">
</div>

### Multi-head attention
- Multi-head attention consists of four parts: **Linear layers**、**Multi-head attention**、**Concatenation of heads** and **Final linear layers**.

<div align='center'>
    <img class="course-image" src="https://github.com/EternalFeather/Transformer-in-generating-dialogue/blob/master/images/multi_head_attention.png">
</div>

### Pointwise Feedforward Network
- Pointwise feedforward network consists of two fully-connected layers with ReLU activation in between.

### Learning Rate Schedule
- Use the adam optimizer with a custom learning rate scheduler according to the formula like:  

<div align='center'>
    <img src="http://latex.codecogs.com/gif.latex?\Large{lrate = d_{model}^{-0.5} * min(step{\_}num^{-0.5}, step{\_}num * warmup{\_}steps^{-1.5})}" />  
</div>

<div align='center'>
    <img class="course-image" src="https://github.com/EternalFeather/Transformer-in-generating-dialogue/blob/master/images/learning_rate.png">
</div>


## Model Downsides
However, such a strong architecture still have some downsides:
- For a time-series, the output for a time-step is calculated from the entire history of only the inputs and current hidden-state(Just like the different between CRF & HMM). So that it may be less efficient.
- As the first part above said, if the input does have a temporal/spatial relationship, like text generation task, the model may be lost in the context.

# Usage
- old_version
    - STEP 1. Download dialogue corpus with format like sample datasets and extract them to `data/` folder.
    - STEP 2. Adjust hyper parameters in `params.py` if you want.
    - STEP 3. Run `make_dic.py` to generate vocabulary files to a new folder named `dictionary`.
    - STEP 4. Run `train.py` to build the model. Checkpoint will be stored in `checkpoint` folder while the tensorflow event files can be found in `logdir`. 
    - STEP 5. Run `eval.py` to evaluate the result with testing data. Result will be stored in `Results` folder.
- new_version(2.0 & 1.12.x with eager mode)
    - follow the .ipynb to run train & eval & demo
        - if you use `GPU` to speed up training processing, please set up your device in the code.(It support multi-workers training)

# Results
- demo
```
- Source: 肥 宅 初 夜 可 以 賣 多 少 `
- Ground Truth: 肥 宅 還 是 去 打 手 槍 吧
- Predict: 肥 宅 還 是 去 打 手 槍 吧

- Source: 兇 的 女 生 484 都 很 胸
- Ground Truth: 我 看 都 是 醜 的 比 較 凶
- Predict: 我 看 都 是 醜 的 比 較 <UNK>

- Source: 留 髮 不 留 頭
- Ground Truth: 還 好 我 早 就 禿 頭 了
- Predict: 還 好 我 早 就 禿 頭 了

- Source: 當 人 好 痛 苦 R 的 八 卦
- Ground Truth: 去 中 國 就 不 用 當 人 了
- Predict: 去 中 國 就 不 會 有 了 -

- Source: 有 沒 有 今 天 捷 運 的 八 卦
- Ground Truth: 有 - 真 的 有 多
- Predict: 有 - 真 的 有 多

- Source: 2016 帶 走 了 什 麼 `
- Ground Truth: HellKitty 麥 當 勞 歡 樂 送 開 門 -
- Predict: <UNK> 麥 當 勞 歡 樂 送 開 門 -

- Source: 有 沒 有 多 益 很 賺 的 八 卦
- Ground Truth: 比 大 型 包 裹 貴
- Predict: 比 大 型 包 <UNK> 貴

- Source: 邊 緣 人 收 到 地 震 警 報 了
- Ground Truth: 都 跑 到 窗 邊 了 才 來
- Predict: 都 跑 到 <UNK> 邊 了 才 來

- Source: 車 震
- Ground Truth: 沒 被 刪 版 主 是 有 眼 睛 der
- Predict: 沒 被 刪 版 主 是 有 眼 睛 der

- Source: 在 家 跌 倒 的 八 卦 `
- Ground Truth: 傷 到 腦 袋 - 可 憐
- Predict: 傷 到 腦 袋 - 可 憐

- Source: 大 家 很 討 厭 核 核 嗎 `
- Ground Truth: 核 核 欠 幹 阿
- Predict: 核 核 欠 幹 阿

- Source: 館 長 跟 黎 明 打 誰 贏 -
- Ground Truth: 我 愛 黎 明 - 我 愛 黎 明 -
- Predict: 我 愛 <UNK> 明 - 我 愛 <UNK> 明 -

- Source: 嘻 嘻 打 打
- Ground Truth: 媽 的 智 障 姆 咪 滾 喇 幹
- Predict: 媽 的 智 障 姆 咪 滾 喇 幹

- Source: 經 典 電 影 台 詞
- Ground Truth: 超 時 空 要 愛 裡 滿 滿 的 梗
- Predict: 超 時 空 要 愛 裡 滿 滿 滿 的

- Source: 2B 守 得 住 街 亭 嗎 `
- Ground Truth: 被 病 毒 滅 亡 真 的 會 -
- Predict: <UNK> 守 得 住
```

# Comparison

## Implement feedforward through fully connected.

- Training Accuracy

<div align='center'>
    <img class="course-image" src="https://i.imgur.com/wZW34e8.png">
</div>

- Training Loss

<div align='center'>
    <img class="course-image" src="https://i.imgur.com/p5MSVVQ.png">
</div>

## Implement feedforward through convolution in only one dimention.

- Training Accuracy

<div align='center'>
    <img class="course-image" src="https://i.imgur.com/y2Q9yM8.png">
</div>

- Training Loss

<div align='center'>
    <img class="course-image" src="https://i.imgur.com/MJdMnvt.png">
</div>

# Tips
&emsp;&emsp;If you try to use **AutoGraph** to speed up your training process, please make sure the datasets is padded to a fixed length. Because of the graph rebuilding operation will be activated during training, which may affect the performance. Our code only ensures the performance of version 2.0, and the lower ones can try to refer it.

# Reference

Thanks for [Transformer](https://github.com/Kyubyong/transformer) and [Tensorflow](https://www.tensorflow.org)
