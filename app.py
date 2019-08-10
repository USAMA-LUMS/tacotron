from flask import Flask, render_template, request
import requests
import argparse
from hparams import hparams, hparams_debug_string
import os
from synthesizer import Synthesizer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/synthesize/<text>')
def synthesize(text):
    url = 'http://127.0.0.1:8080/get_sentence/' + text#request.form['text']
    text = requests.get(url).text
    return synthesizer.synthesize(text)


synthesizer = Synthesizer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    print(hparams_debug_string())
    synthesizer.load(args.checkpoint)
    app.run(host="0.0.0.0", port=args.port, threaded=True)
