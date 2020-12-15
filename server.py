from flask import Flask, url_for, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/classify-news', methods=["POST"])
def classify():
    print("request: ", request)
    return {
        "classificationResult": "True"
    }
