import theano
import theano.tensor as T

import data_io
import model
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates

def data_stream(data_file, word2idx):
    stream = data_io.stream(data_file, word2idx)
    stream = data_io.randomise(stream, buffer_size=300)
    stream = data_io.sortify(stream, lambda x: x[0].shape[0] + x[1].shape[0])
    stream = data_io.batch(stream, batch_size=5)
    stream = data_io.randomise(stream, buffer_size=50)
    stream = data_io.arrayify(stream,
                              start_idx=len(word2idx),
                              end_idx=len(word2idx) + 1)
    stream = data_io.async(stream)
    return stream


if __name__ == "__main__":
    P = Parameters()
    X_1 = T.imatrix('X_1')
    X_2 = T.imatrix('X_2')

    idx2word, word2idx = data_io.load_dictionary('dict.pkl')
    _, cost = model.build(P, embedding_count=len(word2idx) + 2, embedding_size=20)
    recon, kl = cost(X_1.T, X_2.T)
    loss = recon + kl / T.cast(T.sum(X_1.T[1:]), 'float32')
    parameters = P.values()
    gradients = T.grad(loss, wrt=parameters)
    train = theano.function(
        inputs=[X_1, X_2],
        outputs=loss,
        updates=updates.adam(parameters,gradients)
    )
    for batch_1, batch_2 in data_stream('data/train.txt', word2idx):
        print train(batch_1, batch_2)
