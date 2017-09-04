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
    _, decode_step = model.build_decoder(
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
        outputs=latent
    )
    print "Created init function."

    x = T.imatrix('x')
    Z = T.tensor3('Z')

    step = theano.function(
        inputs=[x, Z],
        outputs=decode_step(x.T, Z)
    )
    P.load('val_model.pkl')
    print "Created sampling function."
    # TODO build line reader
    unk_idx = len(word2idx)
    print ">> ",
    for line in fileinput.input():
        line = line.strip()
        if line == "":
            continue
        tokens = np.array([[word2idx.get(w, unk_idx)
                            for w in line.split()]],
                          dtype=np.int32)
        print
        print "Input:", ' '.join(idx2word[idx] for idx in tokens[0])
        print
        print "Outputs:"
        print "--------"
        for i in xrange(2):
            prior_sample = init(tokens)
            choices = np.arange(len(word2idx) + 2)
            idxs = [len(word2idx)]
            for _ in xrange(200):
                probs = step([idxs], prior_sample)
                if i % 2 == 0:
                    idx = np.random.choice(choices, p=probs[0])
                else:
                    idx = np.argmax(probs[0])
                if idx == len(word2idx) + 1:
                    break
                else:
                    print idx2word[idx],
                    idxs.append(idx)
            print
        print ">> ",
