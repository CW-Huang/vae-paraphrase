import theano
import theano.tensor as T
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import data_io
import model


def data_stream(data_file, word2idx):
    stream = data_io.stream(data_file, word2idx)
    stream = data_io.randomise(stream, buffer_size=512)
    stream = data_io.sortify(stream, lambda x: x[0].shape[0],
                             buffer_size=256)
    stream = data_io.batch(stream, batch_size=64)
    stream = data_io.randomise(stream, buffer_size=50)
    stream = data_io.arrayify(stream,
                              start_idx=len(word2idx),
                              end_idx=len(word2idx) + 1)
    stream = data_io.async(stream)
    return stream


if __name__ == "__main__":
    data_location = '/data/lisa/data/sheny/ParaNews/train.txt'
    P = Parameters()
    X_1 = T.imatrix('X_1')
    X_2 = T.imatrix('X_2')

    idx2word, word2idx = data_io.load_dictionary('dict.pkl')
    cost, _, _, _ = model.build(P, embedding_count=len(word2idx) + 2,
                                embedding_size=256)
    recon, kl = cost(X_1.T, X_2.T)
    count = T.cast(T.sum(T.neq(X_1, -1).T[1:]), 'float32')
    loss = (recon + kl) / count
    parameters = P.values()
    gradients = T.grad(loss, wrt=parameters)
    P_train = Parameters()
    train = theano.function(
        inputs=[X_1, X_2],
        outputs=[recon / count, kl / T.cast(X_1.shape[0], 'float32')],
        updates=updates.adam(parameters, gradients,
                             learning_rate=1e-3, P=P_train),
    )

    i = 0
    for batch_1, batch_2 in data_stream(data_location, word2idx):
        print train(batch_1, batch_2)
        i += 1
        if i % 100 == 0:
            print "Saving"
            P.save('model.pkl')
            P_train.save('train.pkl')
            i = 0
