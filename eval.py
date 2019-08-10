import argparse
import os
import re
import requests
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = []


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%03d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      url='http://127.0.0.1:8080/get_sentence/'+text
      text=requests.get(url).text
      f.write(synth.synthesize(text))

def read_sentences():
  with open('test_sentences.txt','r',encoding='utf8') as fin:
    text=fin.read()
    text=text.split('\n')
  for line in text:
    sentences.append(line)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)



if __name__ == '__main__':
  read_sentences()
  main()
