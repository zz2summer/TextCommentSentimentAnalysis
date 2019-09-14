# TextCommentSentimentAnalysis

### 一、项目说明

&emsp;&emsp;**中文文本情感分类:**

&emsp;&emsp;基于深度学习的情感分类和智能客服研究与实现。主要是酒店和书店的评论情感分析，可以判定积极和消极，对于消极评论，还可以判断其具体类别，比如物流不好或者服务差等等。

&emsp;&emsp;项目具体使用说明请参考文件：[./sentimentAnalysis/项目使用说明文档](https://github.com/zz2summer/TextCommentSentimentAnalysis/blob/master/sentimentAnalysis/%E9%A1%B9%E7%9B%AE%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.docx)

&emsp;&emsp;项目开发过程记录参考：[./sentimentAnalysis/实验报告](https://github.com/zz2summer/TextCommentSentimentAnalysis/blob/master/sentimentAnalysis/%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A.docx)

***
### 二、项目基础

&emsp;&emsp;开发环境和开发工具：

        Python 3.7.3，
        TensorFlow 1.14.0，
        Flask 1.1.1，
        PyCharm 2019.1.3 (Professional Edition)，
        JRE: 11.0.2+9-b159.60 amd64，
        JVM: OpenJDK 64-Bit Server VM by JetBrains s.r.o，
        Windows 10 10.0

&emsp;&emsp;开发时间：2019.7.1--2019.7.12

&emsp;&emsp;开发者：Summer

&emsp;&emsp;参考视频：[用tensorflow进行中文自然语言处理中的情感分析](https://www.bilibili.com/video/av30543613)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[源码(用tensorflow进行中文自然语言处理中的情感分析)](https://github.com/zz2summer/chinese_sentiment.git)

***
### 三、演示demo

![demo](https://img-blog.csdnimg.cn/20190914181119163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

![demo2](.\sentimentAnalysis\实验报告iamges\test2.png)

***
### 四、开发流程

**4.1 实现评论的二分类（判断评论为积极还是消极）算法。**

4.1.1 加载数据。

&emsp;&emsp;利用pandas模块的read_excel（）函数对数据（pos.xls、neg.xlsx）进行读取并保存到一个评论list中。

4.1.2 数据上标签。

&emsp;&emsp;二分类算法中积极评论的标签为“1”，消极评论的标签为“0”。

4.1.3 中文分词。

&emsp;&emsp;加载分词向量，使用gensim加载预训练中文分词embedding，本次实验采用的是知乎的分词向量sgns.zhihu.bigram。

4.1.4 提取文本关键词。

4.1.5 建立tokens字典。

&emsp;&emsp;利用先对所得的评论list去除标点，再利用结巴分词对结果进行处理。

4.1.6 使用tokens字典将“文本”转化为“数字列表”，对tokens的分布进行可视化处理，利用matplotlib.pyplot可以实现。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200612632.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

4.1.7 截长补短让所有“数字列表”长度都是一致，取tokens平均值并加上两个tokens的标准差作为最终长度，保证每个文本都是同样的长度可以避免不必要的错误。

4.1.8 Embedding层将“数字列表”转化为向量列表。

4.1.9 将向量列表送入深度学习模型进行训练。

&emsp;&emsp;主要是利用双向和单向LSTM、Dropout、中间层数、注意力机制、激活函数、loss函数、学习率等对模型进行实现和不断优化。

4.1.10 保存模型与模型可视化。

&emsp;&emsp;利用model的save函数将已经训练好的模型进行保存，同时在模型训练中加入Tensorboard的相关函数对过程可视化处理。

&emsp;&emsp;利用matplotlib.pyplot模块可以查看训练过程每一次epoch后的训练精确度和验证精确度的变化过程。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200705419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

&emsp;&emsp;在命令行输入如下命令可以查看训练过程中各项数据的更加详细变化过程。

&emsp;&emsp;tensorboard --logdir=F:\Desktop\sentimentAnalysis\static\logs\class2

&emsp;&emsp;算法的模型结构图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200717999.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

4.1.11 模型的预测功能。

&emsp;&emsp;对输入的文本先进行去除标点符号操作，再进行分词操作、tokenize、padding等，最后调用model.predict（）进行结果预测，得到一个在0-1范围内的浮点数即精确度，大于0.5即偏向积极情感，反之则是偏向消极情感。

4.1.12 训练过程可视化。

&emsp;&emsp;在训练过程中加上Tensorboard相关操作即可记录实时的精确度、loss、learn rate的变化过程。

&emsp;&emsp;训练过程中的train和validation和accuracy的变化如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200728206.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

&emsp;&emsp;训练过程中的loss的对比变化过程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200741214.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

&emsp;&emsp;训练过程中的学习率learn rate的变化过程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200748178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

**4.2 实现对消极评论的九分类即进一步判断消极评论是哪一类的算法。**

&emsp;&emsp;消极评论的九分类和评论的二分类算法基本相似，主要是对标签编码进行独热编码处理和预测结果的调整即可，所以这里主要讲述九分类算法与二分类算法的区别。

4.2.1 数据读取时只读取消极数据，标签根据九个类别分为0-8。

4.2.2 利用tensorflow.python.keras.utils的to_categorical（）函数对标签进行one-hot编码处理。

4.2.3 训练模型对比二分类删去了双向的LSTM，因为根据测试结果对比，本次项目更适合仅使单向的LSTM，其他参数基本不变。

4.2.4 预测结果的是一个一维向量，包含九个比率，利用numpy的argmx（）函数选择其中最大的比率作为最终预测的类别。

4.2.5 训练过程的模型及相关的数据的可视化展示。

&emsp;&emsp;每次epoch后的训练精确度和验证精确度的变化：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200821388.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

&emsp;&emsp;算法的模型结构图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200928941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

&emsp;&emsp;训练过程中的train和validation和accuracy的变化如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200924197.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

&emsp;&emsp;训练过程中的loss的对比变化过程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/201909142009198.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

&emsp;&emsp;训练过程中的学习率learn rate的变化过程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914200913566.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY0NTU1,size_16,color_FFFFFF,t_70)

***
### 五、实验结果

&emsp;&emsp;根据实验的多次模型调整、参数修改与结果优化，二分类和九分类模型的精确度都达到了90%以上，效果相对而言是比较好的。

 &emsp;&emsp;二分类算法模型的精确度可以达到91.73%，
 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914201104276.png)

&emsp;&emsp;而九分类模型的精确度可以达到91.20%，

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190914201108777.png)
