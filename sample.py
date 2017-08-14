import numpy as np
import theano.tensor as T
import theano
from theano_toolkit.parameters import Parameters
import model
import data_io

if __name__ == "__main__":
    data_location = '/data/lisa/data/sheny/ParaNews/train.txt'
    P = Parameters()
    idx2word, word2idx = data_io.load_dictionary('dict.pkl')
    _, initial, sample_prior, sample_step = model.build(
        P,
        embedding_count=len(word2idx) + 2,
        embedding_size=256
    )

    X_1 = T.imatrix('X_1')
    X_2 = T.imatrix('X_2')
    z_sym = sample_prior(X_2.T)
    init = theano.function(
        inputs=[X_2],
        outputs=list(initial(1, z_sym)) + [z_sym]
    )
    x = T.ivector('x')
    z = T.matrix('z')
    prev_cell = T.matrix('prev_cell')
    prev_hidden = T.matrix('prev_hidden')

    step = theano.function(
        inputs=[z, x, prev_cell, prev_hidden],
        outputs=sample_step(z, x, prev_cell, prev_hidden)
    )
    P.load('model.pkl')
    line = ("a u.s. army oh-## helicopter made an emergency landing in north" +
            " korea on saturday , but u.s. officials said they had no confir" +
            "mation of reports out of north korea that the aircraft had been" +
            " shot down .")
    unk_idx = len(word2idx)
    input = np.array([[word2idx.get(w, unk_idx) for w in line.split()]],
                     dtype=np.int32)
    print line
    print
    for _ in xrange(5):
        cell, hidden, prior_sample = init(input)
        choices = np.arange(len(word2idx) + 2)
        idx = len(word2idx)
        while True:
            (probs, cell, hidden) = step(prior_sample, [idx], cell, hidden)
            idx = np.random.choice(choices, p=probs[0])
            if idx == len(word2idx) + 1:
                break
            else:
                print idx2word[idx],
        print
