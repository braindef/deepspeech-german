#!/usr/bin/env bash

set -xe

tuda_corpus_path="../tuda"
voxforge_corpus_path="../voxforge_de"
text_corpus_path="../text_corpora/German_sentences_8mil_filtered_maryfied.txt"

exp_path="../deepspeech-german.output"

kenlm_bin="../kenlm_bin/kenlm/build/bin"
deepspeech="../deepspeech"

# Create LM
./prepare_text.py $text_corpus_path $exp_path/cleaned_vocab.txt

$kenlm_bin/lmplz --text $exp_path/cleaned_vocab.txt --arpa $exp_path/words.arpa --o 3
$kenlm_bin/build_binary -T -s $exp_path/words.arpa  $exp_path/lm.binary

# Create trie
$deepspeech/native_client/generate_trie data/alphabet.txt $exp_path/lm.binary $exp_path/cleaned_vocab.txt $exp_path/trie

# Download/Prepare Data
./prepare_data_tuda_and_voxforge.py $tuda_corpus_path $voxforge_corpus_path $exp_path/data

# Train
./run_training.sh $deepspeech $(realpath data/alphabet.txt) $exp_path

