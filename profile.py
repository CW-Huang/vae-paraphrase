import theano
import theano.tensor as T
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
import data_io
import model
import train as t

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
        profile=True
    )

    i = 0
    for batch_1, batch_2 in t.data_stream(data_location, word2idx):
        print train(batch_1, batch_2)
        i += 1
        if i == 5:
            break
    train.profile.summary()
