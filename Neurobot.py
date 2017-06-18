import re
import numpy as np
import codecs
import os
import json
from tqdm import tqdm


X, y = np.load("batch1.npy"), np.load("batch2.npy")
with open("tokens_id.json") as f:
    tokens_id = json.load(f)
token_to_id = tokens_id[u'token_to_id']
id_to_token = tokens_id["id_to_token"]
tokens = tokens_id["tokens"]
tokens = tokens

print("Deep Learning")
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam
from hierarchical_softmax_layer import HierarchicalSoftmaxDenseLayer


# In[ ]:

input_sequence = T.matrix('token sequence','int32')
target_phonemes = T.matrix('target phonemes','int32')
targets = target_phonemes[:,1:].ravel()
targets = targets.reshape((-1, 1))

mask = T.nonzero(T.neq(targets, -1))


# In[ ]:

l_targets = lasagne.layers.InputLayer((None,1 ), targets)


# In[ ]:


##ENCODER
l_in = lasagne.layers.InputLayer(shape=(None, None),input_var=input_sequence)
l_mask = lasagne.layers.InputLayer(shape=(None, None),input_var=T.neq(input_sequence,-1))
l_emb = lasagne.layers.EmbeddingLayer(l_in, len(tokens), 256)
l_rnn = lasagne.layers.LSTMLayer(l_emb,256,mask_input=l_mask)
l_rnn = lasagne.layers.LSTMLayer(l_rnn,256,only_return_final=True,mask_input=l_mask)

##DECODER
transc_in = lasagne.layers.InputLayer(shape=(None, None),input_var=target_phonemes[:,1:])
transc_mask = lasagne.layers.InputLayer(shape=(None, None),input_var=T.neq(target_phonemes[:,1:],-1))
transc_emb = lasagne.layers.EmbeddingLayer(transc_in, len(tokens), 256)
transc_rnn = lasagne.layers.LSTMLayer(transc_emb,256,hid_init=l_rnn,mask_input=transc_mask)
transc_rnn = lasagne.layers.LSTMLayer(transc_rnn,256,hid_init=l_rnn,mask_input=transc_mask)


#flatten batch and time to be compatible with feedforward layers (will un-flatten later)
transc_rnn_flat = lasagne.layers.reshape(transc_rnn, (-1,transc_rnn.output_shape[-1]))

l_out = HierarchicalSoftmaxDenseLayer(transc_rnn_flat,len(tokens), target=l_targets)



# In[ ]:

weights = lasagne.layers.get_all_params(l_out, trainable=True)
print(weights)


# In[ ]:

network_output_train = lasagne.layers.get_output(l_out)
network_output = lasagne.layers.get_output(l_out, return_probas_anyway=True)
network_output = network_output.reshape([target_phonemes.shape[0], target_phonemes.shape[1], -1])


def crossentropy(answ):
    return -1*T.log(answ).mean()


# In[ ]:

# predictions_flat = network_output[:,:-1,:].reshape([-1, len(tokens)])
targets = target_phonemes[:,1:].ravel()

mask = T.nonzero(T.neq(targets, -1))

loss = crossentropy(network_output_train[mask])
updates = adam(loss, weights)


# Компилируем

# In[ ]:
print("Compile")
t = theano.function([input_sequence, target_phonemes], network_output_train, allow_input_downcast=True)
compute_cost = theano.function([input_sequence, target_phonemes], loss, allow_input_downcast=True)
train = theano.function([input_sequence, target_phonemes], loss, updates=updates, allow_input_downcast=True)


print("Calculate")
print(t(X, y))
print(compute_cost(X, y))
print(train(X, y))

