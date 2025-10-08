# Main application code

import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, Multi-flow Chatbot!'

if __name__ == '__main__':
    app.run(debug=True)
