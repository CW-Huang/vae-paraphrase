import theano
import theano.tensor as T
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import data_io
import model
import itertools
from pprint import pprint
import numpy as np


def data_stream(data_file, word2idx):
    stream = data_io.stream(data_file, word2idx)
    stream = itertools.ifilter(
        lambda x: x[0].shape[0] <= 126 and x[1].shape[0] <= 126,
        stream
    )
    stream = data_io.randomise(stream, buffer_size=512)
    stream = data_io.sortify(stream, lambda x: x[0].shape[0],
                             buffer_size=256)
    stream = data_io.batch(stream, batch_size=16)
    stream = data_io.randomise(stream, buffer_size=50)
    stream = data_io.arrayify2(stream,
                               start_idx=len(word2idx),
                               end_idx=len(word2idx) + 1)
    stream = data_io.async(stream)
    return stream


if __name__ == "__main__":
    data_location = '/data/lisa/data/sheny/ParaNews/train.txt'
    P = Parameters()
    X_12 = T.imatrix('X_12')

    idx2word, word2idx = data_io.load_dictionary('dict.pkl')
    beta = theano.shared(np.float32(0))
    cost = model.build(
        P, embedding_count=len(word2idx) + 2,
        embedding_size=256
    )
    recon, kl = cost(X_12)
    X_1 = X_12[:X_12.shape[0] // 2]
    count = T.cast(T.sum(T.neq(X_1, -1).T[1:]), 'float32')
    loss = (recon + beta * kl) / count
    parameters = P.values()
    pprint(parameters)

    gradients = updates.clip_deltas(T.grad(loss, wrt=parameters), 10)
    P_train = Parameters()
    train = theano.function(
        inputs=[X_12],
        outputs=[recon / count, kl / T.cast(X_12.shape[0] // 2, 'float32')],
        updates=updates.adam(
            parameters, gradients,
            learning_rate=3e-4, P=P_train
        ),
    )

    P.load('model.pkl')
    P_train.load('train.pkl')
    i = 0
    for epoch in xrange(20):
        for batch in data_stream(data_location, word2idx):
            print batch.shape, train(batch)
            i += 1
            if i % 100 == 0:
                print "Saving"
                P.save('model.pkl')
                P_train.save('train.pkl')
                beta_val = beta.get_value()
                print "iteration", i
                print "beta_val", beta_val,
                if i < 1:
                    beta_val = np.float32(0.)
                elif beta_val < np.float32(1):
                    beta_val += np.float32(0.01)

                if beta_val > 1:
                    beta_val = np.float32(1)
                print "new beta_val", beta_val
                beta.set_value(beta_val)
