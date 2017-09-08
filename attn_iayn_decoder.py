import numpy as np
import theano.tensor as T

import feedforward
import transformer


def build_1d_conv(P, name, input_size, output_size, window_size,
                  activation=T.nnet.relu,
                  weight_init=feedforward.relu_init):
    W_val = weight_init(input_size * window_size, output_size)
    P["W_%s" % name] = W_val.reshape((output_size, input_size, 1, window_size))
    b = np.zeros((output_size,))
    P["b_%s" % name] = b
    W = P["W_%s" % name]
    b = P["b_%s" % name].dimshuffle('x', 0, 'x', 'x')

    def conv(X):
        # time x batch_size, hidden_size
        X = X.dimshuffle(1, 2, 'x', 0)
        # batch_size x hidden_size x 1 x sequence_length
        conv_out_ = T.nnet.conv2d(X, W, border_mode='valid') + b
        output = activation(conv_out_)
        output = output.dimshuffle(2, 3, 0, 1)[0]
        return output
    return conv


def build_attention_transform(P, name, q_size, k_size, hidden_size,
                              heads=1, temporal_bias=False):
    P['W_%s_query' % name] = 0.1 * np.random.randn(heads, q_size, hidden_size)
    P['W_%s_key' % name] = 0.1 * np.random.randn(heads, k_size, hidden_size)
    P['b_%s' % name] = np.zeros((heads,))

    W_query = P['W_%s_query' % name]
    W_key = P['W_%s_key' % name]
    b = P['b_%s' % name]

    if temporal_bias:
        P['w_%s_after' % name] = np.zeros((heads,))
        P['w_%s_before' % name] = np.zeros((heads,))
        w_after = P['w_%s_after' % name]
        w_before = P['w_%s_before' % name]

    def attend(query, key, value,
               query_mask=None,
               key_mask=None,
               value_mask=None):
        batch_size = value.shape[0]
        query_length = query.shape[1]
        key_length = key.shape[1]
        # values : batch_size, length, input_size
        query_hidden = T.tensordot(query, W_query, axes=(2, 1))
        key_hidden = T.tensordot(key, W_key, axes=(2, 1))
        # query & keys : batch_size, length, heads, hidden_size
        query_hidden = query_hidden.dimshuffle(0, 2, 1, 3).reshape((
            batch_size * heads, query_length, hidden_size
        ))
        key_hidden = key_hidden.dimshuffle(0, 2, 1, 3).reshape((
            batch_size * heads, key_length, hidden_size
        ))

        # query & keys: batch_size * heads, length, key_size
        time = T.cast(T.arange(key_length), 'float32')
        td = time[None, :] - time[:, None]
        current = T.neq(td, 0)

        if temporal_bias:
            after = (w_after[:, None, None] *
                     T.log(T.switch(td > 0, td, 0) + 1)[None, :, :])
            before = (w_before[:, None, None] *
                      T.log(T.switch(td < 0, -td, 0) + 1)[None, :, :])

        outer_dot = T.batched_tensordot(
            query_hidden, key_hidden, axes=(2, 2)
        ).reshape((
            batch_size, heads, query_length, key_length
        ))
        # outer_dot: batch_size, heads, query_length, key_length
        if temporal_bias:
            overall_mask = current[None, None, :, :]
        else:
            overall_mask = T.ones((
                batch_size, query_length, key_length
            ))[:, None, :, :]
        if query_mask is not None:
            overall_mask = overall_mask and query_mask[:, None, :, None]
        if key_mask is not None:
            overall_mask = overall_mask and key_mask[:, None, None, :]

        attn = transformer.softmax(
            (outer_dot / np.float32(np.sqrt(hidden_size))) +
            (after[None, :, :, :] if temporal_bias else 0) +
            (before[None, :, :, :] if temporal_bias else 0) +
            b[None, :, None, None],
            overall_mask,
            axis=3
        )
        # attn : batch_size, heads, query_length, key_length
        # values : batch_size, key_length, input_size
        output = T.sum(
            attn[:, :, :, :, None] *
            value[:, None, None, :, :], axis=3
        ).dimshuffle(0, 2, 1, 3)
        # output : batch_size, query_length, heads, input_size
        return output
    return attend


def build(P, name, embedding_size, hidden_size, latent_size, context_size=5):
    input_transform = build_1d_conv(
        P, name="%s_input" % name,
        input_size=embedding_size,
        output_size=hidden_size,
        window_size=context_size
    )

    attend = transformer.build_attention_transform(
        P, name="%s_attend" % name,
        q_size=hidden_size,
        k_size=latent_size,
        hidden_size=hidden_size,
        heads=1, temporal_bias=False
    )

    combine = feedforward.build_combine_transform(
        P, name="%s_combine" % name,
        input_sizes=[hidden_size, latent_size],
        output_size=embedding_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu
    )

    def transform(X, latent, mask):
        batch_size = X.shape[1]
        padding = T.alloc(
            P.embedding[-2],
            context_size - 1,
            batch_size,
            embedding_size
        )
        X_ = T.concatenate([padding, X], axis=0)
        hidden_1 = input_transform(X_)
        selected_latent = attend(
            hidden_1.dimshuffle(1, 0, 2),
            latent.dimshuffle(1, 0, 2),
            latent.dimshuffle(1, 0, 2),
            query_mask=mask.dimshuffle(1, 0)
        )
        return combine([
            hidden_1,
            selected_latent[:, :, 0, :].dimshuffle(1, 0, 2)
        ])
    return transform


if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    P.embedding = np.random.randn(5, 20)
    latent = T.as_tensor_variable(
        np.random.randn(5, 6, 16).astype(np.float32)
    )
    X_12 = T.as_tensor_variable(
        np.array([
            [-2,  0,  1,  0,  1,  0,  1,  2, -1, -1, -1],
            [-2,  0,  1,  0,  1,  0,  1,  2,  3,  4, -1],
            [-2,  0,  1,  0,  1,  0,  1,  2,  3, -1, -1],
            [-2,  0,  1,  0,  1,  0,  1,  2,  3,  4,  1],
            [-2,  0,  1,  0,  1,  0,  1,  2,  3,  4,  1],
            [-2,  0,  1,  0,  1,  0,  1,  2,  3,  4,  1]
        ]).astype(np.int32)
    )
    transform = build(
        P, name="test",
        embedding_size=20,
        hidden_size=20,
        latent_size=16,
        context_size=5
    )
    embeddings = P.embedding[X_12.T]
    mask = T.neq(X_12.T, -1)
    print "input shape", embeddings.eval().shape
    hidden = transform(embeddings, latent, mask)
    val = hidden.eval()
    print val.shape
