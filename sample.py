import numpy as np
import theano.tensor as T
import theano
from theano_toolkit.parameters import Parameters
import model
import data_io
import fileinput

if __name__ == "__main__":
    data_location = '/data/lisa/data/sheny/ParaNews/train.txt'
    idx2word, word2idx = data_io.load_dictionary('dict.pkl')
    embedding_count = len(word2idx) + 2
    embedding_size = 256

    P = Parameters()
    P.embedding = np.random.randn(embedding_count, embedding_size)
    _, encoder = model.build_encoder(
        P,
        embedding_size=embedding_size,
        annotation_size=256,
        latent_size=128
    )
    _, initial, decode_step = model.build_decoder(
        P,
        embedding_size=embedding_size,
        embedding_count=embedding_count,
        latent_size=128,
        hidden_size=256
    )

    X_2 = T.imatrix('X_2')
    latent = encoder(P.embedding[X_2.T], T.neq(X_2.T, -1))
    init = theano.function(
        inputs=[X_2],
        outputs=list(initial(latent[0])) + [latent]
    )
    print "Created init function."

    x = T.ivector('x')
    Z = T.tensor3('Z')
    prev_cell = T.matrix('prev_cell')
    prev_hidden = T.matrix('prev_hidden')

    step = theano.function(
        inputs=[x, prev_cell, prev_hidden, Z],
        outputs=decode_step(x, prev_cell, prev_hidden, Z)
    )
    P.load('model.pkl')
    print "Created sampling function."
    # TODO build line reader
    unk_idx = len(word2idx)
    print ">> ",
    for line in fileinput.input():
        line = line.strip()
        tokens = np.array([[word2idx.get(w, unk_idx) for w in line.split()]],
                          dtype=np.int32)
        print
        print "Input:", ' '.join(idx2word[idx] for idx in tokens[0])
        print
        print "Outputs:"
        print "--------"
        for _ in xrange(5):
            cell, hidden, prior_sample = init(tokens)
            choices = np.arange(len(word2idx) + 2)
            idx = len(word2idx)
            for _ in xrange(200):
                (probs, cell, hidden) = step([idx], cell, hidden, prior_sample)
                #idx = np.random.choice(choices, p=probs[0])
                idx = np.argmax(probs[0])
                if idx == len(word2idx) + 1:
                    break
                else:
                    print idx2word[idx],
            print
        print ">> ",
