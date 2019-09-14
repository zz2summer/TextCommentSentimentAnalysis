# model2  91.73
# model9  91.20
import random
# 导入项目运行所需的必要模块等文件
from tensorflow.python.keras import models
from gensim.models import KeyedVectors
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import jieba
import re
import tensorflow as tf
import numpy as np

sess = tf.Session()
graph = tf.get_default_graph()

# 加载已经训练好的二分类和九分类模型
# 目录路径以app.py所在路径为基础
set_session(sess)
model = models.load_model('./static/model/my_model2.h5')
a = np.ones((1, 220))
model.predict(a)

model9 = models.load_model('./static/model/my_model9.h5')
b = np.ones((1, 184))
model9.predict(b)

# 使用gensim加载预训练中文分词embedding, 有可能需要等待1-2分钟
# sgns.zhihu.bigram下载链接
# https://pan.baidu.com/s/1VGOs0RH7DXE5vRrtw6boQA
cn_model = KeyedVectors.load_word2vec_format('./static/embeddings/sgns.zhihu.bigram', binary=False, unicode_errors="ignore")

# 对用户输入的文本进行预处理
def dealText(text, maxLen):
    # print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
            if cut_list[i] >= 50000:
                cut_list[i] = 0
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=maxLen, padding='pre', truncating='pre')
    return tokens_pad


# 对文本向量进行二分类预测
def predict_sentimentType2(tokens_pad):
    # 预测
    global model
    global graph
    with graph.as_default():
        set_session(sess)
        result = model.predict(tokens_pad)
    return int((result[0][0])*100)


# 用户输入及客服反馈信息词典
commentRes = {'inputString': '输入不在系统操作范围哈', 'percent': 0, 'posOrNeg': '你说呢？',
              'commentType': '怎么会有这种吐槽呢？', 'res': '感谢光临小店，一切愉快！'}

# 九分类的具体类型
commentType9 = ['书籍：内容', '书籍：服务、退换货、货不齐全、不给发货', '书籍：质量：旧、掉页、包装、正版',
                '书籍：物流', '书籍：价格', '酒店：服务', '酒店：设施', '酒店：环境', '酒店：价格']


# 对消极的文本向量进行九分类预测
def predict_sentimentType9(tokens_pad):
    # 预测
    global model9
    global graph
    with graph.as_default():
        set_session(sess)
        result = model9.predict(tokens_pad)
    num = np.argmax(result)
    commentRes['commentType'] = commentType9[int(num)]
    commentRes['res'] = negRes[int(num)][random.randint(0, 1)]


# 积极文本对应的客服回复语句
posRes = ['感谢您的反馈，让每一位顾客满意是我们不懈的追求，我们会一如既往的为您提供高品质的服务，期待您的再次光临！！！',
          '非常感谢您对我们做出的评价，感谢您对本店的支持，因为您的赞许让我们充满了动力!我们期待您的再次光临！O(∩_∩)O~',
          '感谢您对我们的支持与厚爱，您的满意就是我们最大的动力，谢谢您对我们的肯定。我们会继续努力的。也请亲们多多支持，多多关照哦。',
          '感谢亲亲的好评，嘻嘻、我们承诺会以更快更好的服务回馈我们的顾客，也期待着您在将来为我们的发展提出宝贵意见。谢谢亲的支持。',
          '感谢亲的好评，授人玫瑰手有余香，您的好评是对我们最大的支持与鼓励，也将是我们不断前进的动力！',
          '谢谢您的光顾，希望与您有更多的合作！再次非常感谢，祝你生活愉快、万事如意。',
          '感谢亲的支持和惠顾，期待下次能够能您提供更优质的服务！如果我们的产品和服务好，请推荐给您的朋友；如果还有什么不满意的，请一定告诉我哦。',
          '非常好的买家 期待与您再次交易 多谢您的支持 么么哒! ',
          '感谢您的光临，您的满意，就是我们最大的安慰，最大的回报，我们加倍努力做得更好！(*^__^*) 嘻嘻……',
          '亲亲，谢谢您的点名表扬，收到了，表示非常开心得意呢，嘿嘿，期待您的下次光临哦。']

# 消极文本对应的客服回复语句
negRes = [['亲，对于您提到的书籍内容问题我们深感抱歉，我们一定会再接再厉不断改进！',
           '亲，对于您提到的书籍内容问题我们深感抱歉，我们一定会再接再厉不断改进！'],
          ['亲，对于您提到的问题我们深感抱歉，我们一定会再接再厉不断改进！',
           '亲，对于您提到的问题我们深感抱歉，我们一定会再接再厉不断改进！'],
          ['亲，我家的宝贝一般质量都是没有问题，可以放心以购买， 有可能某些宝贝在生产过程中出现的误差导致的，我们这边也是感到非常抱歉! 这样吧，您那边退货，邮费我们到付，我们这边给您退款再送您点小礼物，您看这样行吗，希望您可以互相通融， 看中别的款式的我这边也可以帮您优惠点的!',
           '亲，给您带来的不便，深表歉意。您看方便用手机拍个照吗？若是我们的质量问题，必定无条件承担来回运费，给亲退换都成，事后还给亲送上小礼物..........可好？'],
          ['亲，看到您反馈的物流问题了，非常感谢您对我们的理解和包容，我们后续会加强对快递监督，同时本店后期也会加强包装和质检，争取避免此类问题，感谢您的支持与理解。',
           '感谢反馈支持，非常抱歉物流问题带给您不便~小店一定全方位改善！重新测试包装，甄选包装，优化配送！对您的不快再次表示抱歉，您的反馈使小店进步，祝您生活愉快！'],
          ['亲，咱们店铺出售的图书全是正版书籍，在同样主题的情况下，内容上乘，做到了真正的“人无我有，人有我优”。另外，店铺也会不定期举行力度不一的促销活动，选品价格最低5折起。欢迎您关注参与哟，感谢您的支持。',
           '亲，对于您提到的书籍价格问题我们深感抱歉，我们一定会再接再厉不断改进！'],
          ['尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。',
           '尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。'],
          ['尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。',
           '尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。'],
          ['尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。',
           '尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。'],
          ['尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。',
           '尊敬的宾客， 感谢您抽出宝贵的时间给我们点评，您提出的建议我将会转达给相关部门，并尽最大努力做出改善，期待与您的下一次见面。']]


# 根据对文本预测结果的精确度对客服回复语句进行处理
def commentRespose(percent):
    if percent >= 50:
        commentRes['posOrNeg'] = '积极'
        commentRes['commentType'] = '这种吐槽，不介意多来几条！'
        commentRes['res'] = posRes[random.randint(0, 9)]
    else:
        commentRes['posOrNeg'] = '消极'
        predict_sentimentType9(dealText(commentRes['inputString'], 184))