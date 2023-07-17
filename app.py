from flask import Flask, render_template, request
from gensim.models import KeyedVectors

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.variable_start_string = '{{ '
app.jinja_env.variable_end_string = ' }}'
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    # 处理上传的文件，例如保存到本地或者转换为向量等
    return '文件上传成功！'


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    # 处理查询的文本，例如查询相似语句等
    # 这里我们使用了gensim库来查询相似语句
    model = KeyedVectors.load_word2vec_format('path/to/word2vec.bin', binary=True)
    sentences = model.most_similar(query)
    return render_template('index.html', sentences=sentences)


@app.route('/search_')
def search_():
    query = request.args.get('query')
    # results = search_query(query)
    results = None
    return render_template('search.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
