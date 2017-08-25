import numpy as np
import theano
import theano.tensor as T

import feedforward


def softmax(x, mask, axis=-1):
    k = T.max(T.switch(mask, x, -np.inf),
              axis=axis, keepdims=True)
    exp_norm_x = T.switch(mask, T.exp(x - k), 0)
    output = exp_norm_x / T.sum(exp_norm_x, axis=axis, keepdims=True)
    assert(x.ndim == output.ndim)
    return output


def build_self_attention(P, name, input_size, key_size, heads=1):
    P['W_%s_query' % name] = 0 * np.random.randn(heads, input_size, key_size)
    P['W_%s_key' % name] = np.random.randn(heads, input_size, key_size)
    P['w_%s_after' % name] = np.zeros((heads,))
    P['w_%s_before' % name] = np.zeros((heads,))
    P['b_%s' % name] = np.zeros((heads,))

    W_query = P['W_%s_query' % name]
    W_key = P['W_%s_key' % name]
    w_after = P['w_%s_after' % name]
    w_before = P['w_%s_before' % name]
    b = P['b_%s' % name]

    def self_attend(values, mask):
        batch_size = values.shape[0]
        length = values.shape[1]
        # values : batch_size, length, input_size
        mask = mask[:, :, None]
        query = T.tensordot(values, W_query, axes=(2, 1))
        keys = T.tensordot(values, W_key, axes=(2, 1))
        # query & keys : batch_size, length, heads, key_size
        query = query.dimshuffle(0, 2, 1, 3).reshape((
            batch_size * heads, length, key_size
        ))
        keys = keys.dimshuffle(0, 2, 1, 3).reshape((
            batch_size * heads, length, key_size
        ))

        # query & keys: batch_size * heads, length, key_size

        time = T.cast(T.arange(length), 'float32')
        td = time[None, :] - time[:, None]
        current = T.neq(td, 0)
        after = w_after[:, None, None] * T.switch(td > 0, td, 0)[None, :, :]
        before = w_before[:, None, None] * T.switch(td < 0, -td, 0)[None, :, :]
        outer_dot = T.batched_tensordot(
            query, keys, axes=(2, 2)).reshape((
                batch_size, heads, length, length
            ))
        # outer_dot: batch_size, heads, length, length
        attn = softmax(
            (outer_dot / np.float32(np.sqrt(key_size))) +
            after +
            before +
            b[None, :, None, None],
            mask and current,
            axis=-1
        )
        # attn : batch_size, heads, length, length

        output = T.batched_dot(attn, values).dimshuffle(0, 2, 1, 3)
        # output : batch_size, length, heads, input_size
        return output
    return self_attend

def build_layer(P, name, input_size, output_size,
                key_size, hidden_size, heads=1):
    self_attend = build_self_attention(
        P, name="%s_attn" % name,
        input_size=input_size,
        key_size=key_size,
        heads=heads
    )

    layer_transform = feedforward.build_classifier(
        P, name="%s_transform" % name,
        input_sizes=[input_size, input_size * heads],
        hidden_sizes=[hidden_size],
        output_size=output_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        output_activation=lambda x: x,
        output_weights=feedforward.relu_init
    )

    def transform(X, mask=None):
        if mask is None:
            mask = T.ones_like(X[:, :, 0])

        selected = self_attend(X, mask)
        selected = selected.reshape((
            selected.shape[0],
            selected.shape[1], -1
        ))

        _, output = layer_transform([X, selected])
        output = T.switch(mask[:, :, None], output, 0)
        return output
    return transform


if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    transform = build_layer(
        P, name="test",
        input_size=10,
        key_size=11,
        hidden_size=12,
        output_size=13,
    )
    X = T.as_tensor_variable(
        np.random.randn(2, 20, 10)
    )
    print P.values()
    print transform(X).eval().shape

