import numpy as np
import theano.tensor as T
import theano
from theano_toolkit.parameters import Parameters
import model_conv_attn
import data_io

if __name__ == "__main__":
    P = Parameters()
    idx2word, word2idx = data_io.load_dictionary('dict.pkl')
    _, encode_to_latent, initial, step = model_conv_attn.build(
        P,
        embedding_count=len(word2idx) + 2,
        embedding_size=256
    )

    X_2 = T.imatrix('X_2')
    embeddings = P.embedding[X_2]

    Z, _, _ = encode_to_latent(
        embeddings,
        T.ones_like(embeddings[:, :, 0])
    )
    init = theano.function(
        inputs=[X_2],
        outputs=list(initial(1, Z)) + [Z]
    )

    x = T.ivector('x')
    Z_ = T.tensor3('Z_')
    prev_cell = T.matrix('prev_cell')
    prev_hidden = T.matrix('prev_hidden')

    do_step = theano.function(
        inputs=[x, prev_cell, prev_hidden, Z_],
        outputs=step(x, prev_cell, prev_hidden,
                     T.ones_like(Z_[:, :, 0]), Z_)
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
            (probs, cell, hidden) = do_step([idx], cell, hidden, prior_sample)
            idx = np.random.choice(choices, p=probs[0])
            if idx == len(word2idx) + 1:
                break
            else:
                print idx2word[idx],
        print
