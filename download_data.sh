#!/bin/bash
# Fetch model weights, tokenizer and corpora from public repositories.
set -e
mkdir -p models data && cd models
curl -sSL -o gpt2-10.onnx https://media.githubusercontent.com/media/onnx/models/main/validated/text/machine_comprehension/gpt-2/model/gpt2-10.onnx
curl -sSL -o encoder.json https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/master/GPT2/encoder.json
curl -sSL -o vocab.bpe https://raw.githubusercontent.com/graykode/gpt-2-Pytorch/master/GPT2/vocab.bpe
cd ../data
for c in reuters twitter_samples; do
  curl -sSL -o $c.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/$c.zip && unzip -oq $c.zip
done
curl -sSL -o ag_train.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
curl -sSL -o ag_test.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv
for f in TREC.train.all TREC.test.all custrev.all rt-polarity.all mpqa.all; do
  curl -sSL -o $f https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/$f
done
