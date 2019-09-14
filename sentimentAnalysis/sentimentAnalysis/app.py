from flask import Flask, render_template, request
from src.useModel import predict_sentimentType2, dealText, commentRespose, commentRes, graph

app = Flask(__name__)


# 项目启动时，跳转到用户交互界面的首页
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return render_template('welcome.html')


# 在首页点击进入项目时，跳转到用户交互界面的操作页
@app.route('/showPage', methods=['GET', 'POST'])
def showPage():
    return render_template('showPage.html')

# 在操作页面中点击提交后，获取url中用户输入的文本并调用相关函数进行处理，
# 之后将得到的结果再反馈到界面，显示给用户
@app.route('/result', methods=['GET', 'POST'])
def result():
    commentRes['inputString'] = request.args.get('inputString')
    # n = predict_sentiment("酒店很好")
    with graph.as_default():
        commentRes['percent'] = predict_sentimentType2(dealText(commentRes['inputString'], 220))
    commentRespose(commentRes['percent'])
    print(commentRes['percent'])
    return render_template('showPage.html', inputString=commentRes['inputString'],
                           percent=commentRes['percent'], posOrNeg=commentRes['posOrNeg'],
                           commentType=commentRes['commentType'], res=commentRes['res'])


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
