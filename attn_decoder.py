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


def build(P, name, embedding_size, annotation_size,
          hidden_size=256,
          attn_hidden_size=128):
    lstm_step, non_sequences = lstm.build_step(
        P, name="%s_attn_decoder" % name,
        input_sizes=[embedding_size, annotation_size],
        hidden_size=hidden_size,
    )

    P['W_%s_init_hidden' % name] = feedforward.initial_weights(annotation_size, hidden_size)
    P['b_%s_init_hidden' % name] = np.zeros((hidden_size,))
    P['W_%s_init_cell' % name] = feedforward.initial_weights(annotation_size, hidden_size)
    P['b_%s_init_cell' % name] = np.zeros((hidden_size,))


    W_init_hidden = P['W_%s_init_hidden' % name]
    b_init_hidden = P['b_%s_init_hidden' % name]
    W_init_cell = P['W_%s_init_cell' % name]
    b_init_cell = P['b_%s_init_cell' % name]


    W_val = feedforward.initial_weights(
        input_size=hidden_size + embedding_size + annotation_size,
        output_size=attn_hidden_size
    )
    acc = 0
    P['W_%s_hidden_attn_hidden' % name] = W_val[acc:acc + hidden_size]
    acc += hidden_size
    P['W_%s_embedding_attn_hidden' % name] = W_val[acc:acc + embedding_size]
    acc += embedding_size
    P['W_%s_annotation_attn_hidden' % name] = W_val[acc:acc + annotation_size]
    P['b_%s_attn_hidden' % name] = np.zeros((attn_hidden_size,))
    P['w_%s_attn_out' % name] = np.zeros((attn_hidden_size,))

    W_h_ah = P['W_%s_hidden_attn_hidden' % name]
    W_e_ah = P['W_%s_embedding_attn_hidden' % name]
    W_l_ah = P['W_%s_annotation_attn_hidden' % name]
    b_ah = P['b_%s_attn_hidden' % name]
    w_a = P['w_%s_attn_out' % name]

    def _step(mask_dest, embedding,
              prev_cell, prev_hidden,
              mask_src, annotation,
              W_h_ah, W_e_ah, W_l_ah, b_ah, w_a,
              *non_seq):
        # prev_hidden: batch_size x hidden_size
        # annotation: seq_length x batch_size x feat_size
        # embedding: batch_size x embedding_size
        z = T.dot(embedding[None, :, :], W_e_ah)
        z += T.dot(prev_hidden[None, :, :], W_h_ah)
        z += T.dot(annotation, W_l_ah)
        attn_hidden = T.tanh(
            z + b_ah
        )

        # TODO use tensordot
        attn = softmax(T.dot(attn_hidden, w_a), mask_src,
                       axis=0)[:, :, None]
        # attn_hidden_size x batch_size

        context = T.sum(annotation * attn, axis=0)

        cell, hidden = lstm_step(embedding, context, prev_cell, prev_hidden,
                                 *non_seq)
        cell = T.switch(mask_dest, cell, prev_cell)
        hidden = T.switch(mask_dest, hidden, prev_hidden)
        return cell, hidden

    def initial(annotation):
        first_slot = annotation[0, :]
        init_hidden_batch = T.tanh(T.dot(first_slot, W_init_hidden) +
                                   b_init_hidden)
        init_cell_batch = (T.dot(first_slot, W_init_cell) +
                           b_init_cell)
        return (init_cell_batch,
                init_hidden_batch)

    def decode(mask_dst, mask_src, embeddings, annotation):
        [cells, hiddens], _ = theano.scan(
            _step,
            sequences=[mask_dst[:, :, None], embeddings],
            outputs_info=initial(annotation),
            non_sequences=[
                mask_src, annotation,
                W_h_ah, W_e_ah, W_l_ah, b_ah, w_a] + non_sequences,
            strict=True
        )
        return hiddens

    def step(embedding, prev_cell, prev_hidden, mask_src, annotation):
        return _step(
            [1], embedding,
            prev_cell, prev_hidden,
            mask_src, annotation,
            W_h_ah, W_e_ah, W_l_ah, b_ah, w_a,
            *non_sequences
        )

    return decode, initial, step

if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    embedding_size = 20
    annotation_size = 30
    decode, _, _ = build_decoder(P, embedding_size=embedding_size, annotation_size=annotation_size)

    latents = T.as_tensor_variable(
        np.random.randn(1, 100, annotation_size).astype(np.float32))
    embeddings = T.as_tensor_variable(
        np.random.randn(1, 100, embedding_size).astype(np.float32))
    mask = T.ones_like(embeddings[:, :, 0])
    hiddens = decode(mask, mask, embeddings, latents)
    h_val = hiddens.eval()
    print h_val.shape


