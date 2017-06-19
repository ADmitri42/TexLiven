# coding: utf-8

import re
import numpy as np
import json
import theano
import theano.tensor as T
import lasagne

from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam

import logging

logger = logging.getLogger("TexLiven/network")
hdlr = logging.FileHandler('TexLivenMind.log', encoding = "UTF-8")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def as_matrix(sequences,token_to_i, max_len=None):
    max_len = max_len or max(map(len,sequences))

    matrix = np.zeros((len(sequences),max_len),dtype='int16') -1
    for i,seq in enumerate(sequences):
        row_ix = list(filter(None.__ne__, map(token_to_i.get,seq)))[:max_len]
        matrix[i,:len(row_ix)] = row_ix

    return matrix

from apply_bpe import BPE
with open("codes.txt") as fp:
    bpe = BPE(fp)

def apply_bpe(text):
    text = bpe.segment(text)
    text = re.sub(r"@@ ", "\@/", text)
    return re.sub(r" ", "\@/ \@/", text)


class Texliven(object):
    def __init__(self, pretrain=True, file_with_weights="Networks_weights.npz"):
        logger.info("Get tokens")
        with open("tokens_id.json") as f:
            tokens_id = json.load(f)

        self.token_to_id = tokens_id[u'token_to_id']
        self.id_to_token = tokens_id["id_to_token"]
        tokens = tokens_id["tokens"]
        self.tokens = tokens

        logger.info("Build network")
        input_sequence = T.matrix('token sequence', 'int32')
        target_phonemes = T.matrix('target phonemes', 'int32')


        ##ENCODER
        l_in = lasagne.layers.InputLayer(shape=(None, None),input_var=input_sequence)
        l_mask = lasagne.layers.InputLayer(shape=(None, None),input_var=T.neq(input_sequence,-1))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, len(tokens), 256)
        l_rnn = lasagne.layers.LSTMLayer(l_emb,256,mask_input=l_mask)
        l_rnn = lasagne.layers.LSTMLayer(l_rnn,256,only_return_final=True,mask_input=l_mask)

        ##DECODER
        transc_in = lasagne.layers.InputLayer(shape=(None, None),input_var=target_phonemes)
        transc_mask = lasagne.layers.InputLayer(shape=(None, None),input_var=T.neq(target_phonemes,-1))
        transc_emb = lasagne.layers.EmbeddingLayer(transc_in, len(tokens), 50)
        transc_rnn = lasagne.layers.LSTMLayer(transc_emb,256,hid_init=l_rnn,mask_input=transc_mask)
        transc_rnn = lasagne.layers.LSTMLayer(transc_rnn,256,hid_init=l_rnn,mask_input=transc_mask)


        #flatten batch and time to be compatible with feedforward layers (will un-flatten later)
        transc_rnn_flat = lasagne.layers.reshape(transc_rnn, (-1,transc_rnn.output_shape[-1]))

        l_out = lasagne.layers.DenseLayer(transc_rnn_flat,len(tokens),nonlinearity=lasagne.nonlinearities.softmax)


        self.l_out = l_out

        weights = lasagne.layers.get_all_params(l_out, trainable=True)

        network_output = lasagne.layers.get_output(l_out)
        # In[18]:

        network_output = network_output.reshape([target_phonemes.shape[0], target_phonemes.shape[1], -1])

        predictions_flat = network_output[:, :-1, :].reshape([-1, len(tokens)])
        targets = target_phonemes[:, 1:].ravel()

        mask = T.nonzero(T.neq(targets, -1))

        loss = categorical_crossentropy(predictions_flat[mask], targets[mask]).mean()
        updates = adam(loss, weights)

        logger.info("Compile network")

        self.train = theano.function([input_sequence, target_phonemes], loss, updates=updates, allow_input_downcast=True)

        network_output = network_output.reshape((target_phonemes.shape[0], target_phonemes.shape[1], len(tokens)))
        # predictions for next tokens (after sequence end)
        last_word_probas = network_output[:, -1]
        self.probs = theano.function([input_sequence, target_phonemes], last_word_probas, allow_input_downcast=True)

        if pretrain:
            logger.info("Load weights")
            with np.load(file_with_weights, encoding="bytes") as weights_file:
                lasagne.layers.set_all_param_values(l_out, weights_file["arr_0"])
        logger.info("Ready to use")

    def generate_answer(self, question,answer_prefix = ("START",),t=1,sample=True, max_len=40):
    
        answer = list(answer_prefix)
        question = question.lower()
        question = ["START"] + re.split(r"\\@/", apply_bpe(question)) + ["END"]
        for _ in range(max_len):
    #         print(as_matrix([question],token_to_id))
            if len(answer) < 2:
                answ_matrix = as_matrix([answer],self.token_to_id, max_len=2)
            else:
                answ_matrix = as_matrix([answer],self.token_to_id)
    #         print(answ_matrix)
            next_let_probs = self.probs(as_matrix([question],self.token_to_id), answ_matrix).ravel()
            next_let_probs = next_let_probs**t / np.sum(next_let_probs**t)

            if sample:
                next_letter = np.random.choice(self.tokens,p=next_let_probs) 
            else:
                next_letter = self.tokens[np.argmax(next_let_probs)]

            answer.append(next_letter)

            if next_letter=="END":
                break
        return "".join(answer[1:-1])