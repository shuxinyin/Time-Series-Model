# 时序模型多步预测（Time Series Forecasting Based on Pytorch）

## Introduction

本项目主要基于Pytorch, 验证常见的Time Series Forecasting模型在不同中文时序数据集上的表现。
Time Series Forecasting系列模型实践，包括如下：

1. TFT
2. informer
3. PatchTST
4. ...

### Dataset Introduction

mainly tested on ner dataset as below:  
时序数据集：

关于时序数据处理成以下格式:

```yaml

```

## Environment

python==3.8、torch==2.0.1、scikit-learn=1.3.0  
Or run the shell

```
pip install -r requirements.txt
```

## Project Structure

- config：some model parameters define
- datasets：数据管道
- losses:损失函数
- metrics:评价指标
- models:存放自己实现的BERT模型代码
- output:输出目录,存放模型、训练日志
- processors:数据处理
- script：脚本
- utils: 工具类
- train.py: 主函数

## Usage

### Quick Start

you can start training model by run the shell

1. run ner model except mrc model

```
bash script/train.sh
```


### Results

top F1 score of results on test：

| model/f1_score | Stallion | Electricity |
|----------------|----------|-------------|
| TST            | ~        | ~           |

## Paper & Refer

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf)
- [Informer: Beyond Efficient Transformer for Long Sequence](https://arxiv.org/abs/2012.07436)
- [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.](https://arxiv.org/abs/2211.14730)
- [tft code refer](https://github.com/PlaytikaOSS/tft-torch)







