"""a minimalist flask app"""
from flask import Flask

# note that this name `app` is significant as it is used by twistd
app = Flask(__name__)


@app.route('/')
def hello_world():
    """this is"""
    return 'Hello, World!'
