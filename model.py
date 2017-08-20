import numpy as np
import lstm
import theano.tensor as T
import theano
import vae
import theano_toolkit.utils as U
import feedforward


def softmax(x, axis=-1):
    k = T.max(x, axis=axis, keepdims=True)
    exp_norm_x = T.exp(x - k)
    return exp_norm_x / T.sum(exp_norm_x, axis=axis, keepdims=True)


def build_bilstm(P, input_size, hidden_size):
    lstm_step_f, non_sequences_f = lstm.build_step(
        P, name="lstm_forward",
        input_sizes=[input_size],
        hidden_size=hidden_size,
    )

    lstm_step_b, non_sequences_b = lstm.build_step(
        P, name="lstm_backward",
        input_sizes=[input_size],
        hidden_size=hidden_size,
    )

    def _step(mask_f, mask_b,
              embedding_f, embedding_b,
              prev_cell_f, prev_hidden_f,
              prev_cell_b, prev_hidden_b, *non_sequences):
        f_non_seq = non_sequences[:len(non_sequences_f)]
        b_non_seq = non_sequences[len(non_sequences_f):]
        cell_f, hidden_f = lstm_step_f(embedding_f, prev_cell_f, prev_hidden_f,
                                       *f_non_seq)
        cell_b, hidden_b = lstm_step_b(embedding_b, prev_cell_b, prev_hidden_b,
                                       *b_non_seq)
        cell_f = T.switch(mask_f, cell_f, prev_cell_f)
        cell_b = T.switch(mask_b, cell_b, prev_cell_b)
        hidden_f = T.switch(mask_f, hidden_f, prev_hidden_f)
        hidden_b = T.switch(mask_b, hidden_b, prev_hidden_b)
        return cell_f, hidden_f, cell_b, hidden_b

    def initial(batch_size):
        init_hidden = T.zeros((hidden_size,))
        init_cell = T.zeros((hidden_size,))

        init_hidden = T.tanh(init_hidden)
        init_cell = init_cell
        init_hidden_batch = T.alloc(init_hidden, batch_size, hidden_size)
        init_cell_batch = T.alloc(init_cell, batch_size, hidden_size)
        return (init_cell_batch,
                init_hidden_batch)

    def process(X, mask):
        (init_cell_batch,
         init_hidden_batch) = initial(X.shape[1])
        mask_f = mask[:, :, None]
        mask_b = mask_f[::-1]
        X_f = X
        X_b = X[::-1]
        [cells_f, hiddens_f, cells_b, hiddens_b], _ = theano.scan(
            _step,
            sequences=[mask_f, mask_b, X_f, X_b],
            outputs_info=[init_cell_batch,
                          init_hidden_batch,
                          init_cell_batch,
                          init_hidden_batch],
            non_sequences=non_sequences_f + non_sequences_b,
            strict=True,
        )
        return cells_f, hiddens_f, cells_b[::-1], hiddens_b[::-1]
    return process


def build_encoder(P, hidden_size, embedding_size, latent_size):
    bilstm = build_bilstm(P, embedding_size, hidden_size)
    P.W_encode_transform_f = np.random.randn(hidden_size, latent_size)
    P.W_encode_transform_b = np.random.randn(hidden_size, latent_size)
    gaussian_output = vae.build_encoder_output(
        P, name="enc_out",
        input_size=latent_size,
        output_size=latent_size,
        initialise_weights=lambda x, y: 0.01 * np.random.randn(x, y)
    )

    def encode(embeddings, mask):
        _, hiddens_f, _, hiddens_b = bilstm(embeddings, mask)
        annotation = (T.dot(hiddens_f, P.W_encode_transform_f) +
                      T.dot(hiddens_b, P.W_encode_transform_b))
        mask = mask[:, :, None]
        pool = T.max(T.switch(mask, annotation, -np.inf), axis=0)
        return gaussian_output(pool)
    return encode


def build_decoder(P, embedding_count, embedding_size, latent_size):
    hidden_size = embedding_size
    lstm_step, non_sequences = lstm.build_step(
        P, name="lstm_decoder",
        input_sizes=[embedding_size, latent_size],
        hidden_size=hidden_size,
    )
    P.b_decoder_output = np.zeros((embedding_count,))

    P.W_init_hidden = feedforward.initial_weights(latent_size, embedding_size)
    P.W_init_cell = feedforward.initial_weights(latent_size, embedding_size)

    def _step(mask, embedding,
              prev_cell, prev_hidden,
              latent, *non_seq):
        cell, hidden = lstm_step(embedding, latent, prev_cell, prev_hidden,
                                 *non_seq)
        cell = T.switch(mask, cell, prev_cell)
        hidden = T.switch(mask, hidden, prev_hidden)
        return cell, hidden

    def sample_step(mask, embedding, prev_cell, prev_hidden, latent):
        return _step(mask, embedding, prev_cell, prev_hidden, latent,
                     *non_sequences)

    def initial(batch_size, latent):
        init_hidden = T.dot(latent, P.W_init_hidden)
        init_cell = T.dot(latent, P.W_init_cell)

        init_hidden = T.tanh(init_hidden)
        init_cell = init_cell
        init_hidden_batch = T.alloc(init_hidden, batch_size, hidden_size)
        init_cell_batch = T.alloc(init_cell, batch_size, hidden_size)
        return (init_cell_batch,
                init_hidden_batch)

    def decode(mask, embeddings, latent):
        mask = mask[:, :, None]
        [cells, hiddens], _ = theano.scan(
            _step,
            sequences=[mask, embeddings],
            outputs_info=initial(embeddings.shape[1], latent),
            non_sequences=[latent] + non_sequences,
            strict=True,
        )

        lin_output = T.dot(hiddens, P.embedding.T) + P.b_decoder_output
        return lin_output
    return decode, initial, sample_step


def build(P, embedding_size, embedding_count):
    hidden_size = 256
    latent_size = 256
    P.embedding = np.random.randn(embedding_count,
                                  embedding_size)

    encode = build_encoder(P, hidden_size, embedding_size, latent_size)
    decode, initial, sample_step_ = build_decoder(P, embedding_count,
                                                 embedding_size, latent_size)

    def encode_decode(X_1):
        mask_1 = T.neq(X_1, -1)
        embeddings_1 = P.embedding[X_1]
        z_sample, z_mean, z_std = encode(embeddings_1, mask_1)
        lin_output = decode(mask_1[:-1], embeddings_1[:-1], z_sample)
        return lin_output, z_mean, z_std

    def prior(X_2):
        mask_2 = T.neq(X_2, -1)
        embeddings_2 = P.embedding[X_2]
        z_prior_sample, z_prior_mean, z_prior_std = encode(
            embeddings_2, mask_2)
        return z_prior_sample, z_prior_mean, z_prior_std

    def cost(X_1, X_2):
        _, z_prior_mean, z_prior_std = prior(X_2)
        lin_output, z_mean, z_std = encode_decode(X_1)
        kl = T.sum(vae.kl_divergence(z_mean, z_std, z_prior_mean, z_prior_std),
                   axis=0)
        recon = T.sum(recon_cost(lin_output, X_1[1:]))
        return recon, kl

    def sample_prior(X_2):
        z_prior_sample, _, _ = prior(X_2)
        return z_prior_sample

    def sample_step(latent, x, prev_cell, prev_hidden):
        cell, hidden = sample_step_(
            [1], P.embedding[x],
            prev_cell, prev_hidden,
            latent
        )
        probs = T.nnet.softmax(
            T.dot(hidden, P.embedding.T) + P.b_decoder_output
        )
        return probs, cell, hidden
    return cost, initial, sample_prior, sample_step


def recon_cost(output_lin, labels):
    output = T.nnet.softmax(
        output_lin.reshape((
            output_lin.shape[0] * output_lin.shape[1],
            output_lin.shape[2]
        ))
    )

    labels = labels.reshape((labels.shape[0] * labels.shape[1],))
    mask = T.neq(labels, -1)
    labels = T.switch(mask, labels, 0)
    labels = T.cast(labels, 'int32')

    xent = T.switch(mask,
                    T.nnet.categorical_crossentropy(output, labels), 0)
    return xent


if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    _, cost = build(P, 10, 20)
    recon, kl = cost(
        np.array([[0, 1, 2, -1, -1],
                  [0, 1, 2, 3, 4]]).astype(np.int32).T,
        np.array([[0, 1, 2, 3, -1, -1],
                  [0, 1, 2, 3, 4, 1]]).astype(np.int32).T
    )


