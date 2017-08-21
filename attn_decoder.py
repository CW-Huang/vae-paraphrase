import numpy as np
import lstm
import theano.tensor as T
import theano
import feedforward


def softmax(x, mask, axis=-1):
    k = T.max(T.switch(mask, x, -np.inf),
              axis=axis, keepdims=True)
    exp_norm_x = T.switch(mask, T.exp(x - k), 0)
    output = exp_norm_x / T.sum(exp_norm_x, axis=axis, keepdims=True)
    assert(x.ndim == output.ndim)
    return output


def build_decoder(P, embedding_size, latent_size):
    hidden_size = embedding_size
    print embedding_size, latent_size
    lstm_step, non_sequences = lstm.build_step(
        P, name="lstm_decoder",
        input_sizes=[embedding_size, latent_size],
        hidden_size=hidden_size,
    )

    P.W_init_hidden = feedforward.initial_weights(latent_size, hidden_size)
    P.b_init_hidden = np.zeros((hidden_size,))
    P.W_init_cell = feedforward.initial_weights(latent_size, hidden_size)
    P.b_init_cell = np.zeros((hidden_size,))

    W_val = feedforward.initial_weights(
        input_size=hidden_size + embedding_size + latent_size,
        output_size=128
    )
    acc = 0
    P.W_hidden_attn_hidden = W_val[acc:acc + hidden_size]
    acc += hidden_size
    P.W_embedding_attn_hidden = W_val[acc:acc + embedding_size]
    acc += embedding_size
    P.W_latent_attn_hidden = W_val[acc:acc + latent_size]
    P.b_attn_hidden = np.zeros((128,))

    P.w_attn_out = np.zeros((128,))

    W_h_ah = P.W_hidden_attn_hidden
    W_e_ah = P.W_embedding_attn_hidden
    W_l_ah = P.W_latent_attn_hidden
    b_ah = P.b_attn_hidden
    w_a = P.w_attn_out

    def _step(mask_dest, embedding,
              prev_cell, prev_hidden,
              mask_src, latent,
              W_h_ah, W_e_ah, W_l_ah, b_ah, w_a,
              *non_seq):
        # prev_hidden: batch_size x hidden_size
        # latent : batch_size x seq_length x feat_size
        # embedding: batch_size x embedding_size

        attn_hidden = T.tanh(
            T.dot(embedding[:, None, :], W_e_ah) +
            T.dot(prev_hidden[:, None, :], W_h_ah) +
            T.dot(latent, W_l_ah) +
            b_ah
        )
        attn = softmax(T.dot(attn_hidden, w_a), mask_src)[:, :, None]
        # batch_size x attn_hidden_size

        context = T.sum(latent * attn, axis=1)

        cell, hidden = lstm_step(embedding, context, prev_cell, prev_hidden,
                                 *non_seq)
        cell = T.switch(mask_dest, cell, prev_cell)
        hidden = T.switch(mask_dest, hidden, prev_hidden)
        return cell, hidden

    def initial(batch_size, latent):
        first_slot = latent[:, 0]
        init_hidden_batch = T.tanh(T.dot(first_slot, P.W_init_hidden) +
                                   P.b_init_hidden)
        init_cell_batch = (T.dot(first_slot, P.W_init_cell) +
                           P.b_init_cell)
        return (init_cell_batch,
                init_hidden_batch)

    def decode(mask_dst, mask_src, embeddings, latent):
        embeddings = embeddings.dimshuffle(1, 0, 2)
        mask_dst = mask_dst.dimshuffle(1, 0, 'x')
        [cells, hiddens], _ = theano.scan(
            _step,
            sequences=[mask_dst, embeddings],
            outputs_info=initial(embeddings.shape[1], latent),
            non_sequences=[
                mask_src, latent,
                W_h_ah, W_e_ah, W_l_ah, b_ah, w_a] + non_sequences,
            strict=True
        )
        return hiddens

    def step(embedding, prev_cell, prev_hidden, mask_src, latent):
        return _step(
            [1], embedding,
            prev_cell, prev_hidden,
            mask_src, latent,
            W_h_ah, W_e_ah, W_l_ah, b_ah, w_a,
            *non_sequences
        )

    return decode, initial, step

if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    embedding_size = 20
    latent_size = 30
    decode, _, _ = build_decoder(P, embedding_size=embedding_size, latent_size=latent_size)

    latents = T.as_tensor_variable(
        np.random.randn(1, 100, latent_size).astype(np.float32))
    embeddings = T.as_tensor_variable(
        np.random.randn(1, 100, embedding_size).astype(np.float32))
    mask = T.ones_like(embeddings[:, :, 0])
    hiddens = decode(mask, mask, embeddings, latents)
    h_val = hiddens.eval()
    print h_val.shape


