An Implementation of Attention is all you need with Chinese Corpus
===
# Requirements
- Numpy >= 1.13.1
- Tensorflow-gpu >= 1.2.1
- tqdm
- nltk

# Construction Details
As we all know Translation System can be used in implementing conversational model just by replacing the paris of two different sentences to questions and answers. After all, the basic conversation model named "Sequence-to-Sequence" is develped from translation system. Therefore, why we not to improve the efficiency of conversation model in generating dialogues?
![](https://i.imgur.com/x5FRdRo.png)

This is the structure of transformer which is the core of implementing our model. Now let's split it into several points:

- First one is Input Datasets(Get the batch datasets from generator, which is represented as a list of token ids in this experiment).
- Second one is Embedding layers(Including two parts:**Dataset Embedding** and **Positional Embedding**)
    - Dataset Embedding transform input token ids into a one-hot vector whose size is the length of vocabulary.
    - Positional Embedding also called positional encoding. It considered the index of each word in the list of sentence as the position symbol.
    - Third we have a multi-head attention model to split the output of embedding layers into many pieces and run through different attention models parallelly. Finally we can get the result by concating all the outputs from every models.
    - Finally, going through a feed forward layer and combining with residual items, so that we can get the result. 

![](https://i.imgur.com/YfKUgIC.png)

# Usage
- STEP 1. Download dialogue corpus with format like sample datasets and extract them to `data/` folder.
- STEP 2. Adjust hyper parameters in `params.py` if you want.
- STEP 3. Run `make_dic.py` to generate vocabulary files to a new folder named `dictionary`.
- STEP 4. Run `train.py` to build the model. Checkpoint will be stored in `checkpoint` folder while the tensorflow event files can be found in `logdir`. 
- STEP 5. Run `eval.py` to evaluate the result with testing data. Result will be stored in `Results` folder.

# Results
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

![](https://i.imgur.com/wZW34e8.png)

- Training Loss

![](https://i.imgur.com/p5MSVVQ.png)

## Implement feedforward through convolution in only one dimention.

- Training Accuracy

![](https://i.imgur.com/y2Q9yM8.png)

- Training Loss

![](https://i.imgur.com/MJdMnvt.png)

# Reference

Thanks for [Transformer](https://github.com/Kyubyong/transformer)

- 本文在原先的模组上添加了Attention is all you need提到的Position encoding的部分
