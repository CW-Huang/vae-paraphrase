import numpy as np
import theano.tensor as T

import feedforward
import tracker


def softmax(x, mask, axis=-1):
    k = T.max(x,  # T.switch(mask, x, -np.inf),
              axis=axis, keepdims=True)
    exp_norm_x = T.switch(mask, T.exp(x - k), 0)
    sum_exp = T.sum(exp_norm_x, axis=axis, keepdims=True)
    output = exp_norm_x / sum_exp
    output = T.switch(mask, output, 0)
    assert(x.ndim == output.ndim)
    return output


def build_attention_transform(P, name, q_size, k_size, hidden_size,
                              heads=1, temporal_bias=False,
                              generation_mask=False):
    P['W_%s_query' % name] = 0.3 * np.random.randn(heads, q_size, hidden_size)
    P['W_%s_key' % name] = 0.3 * np.random.randn(heads, k_size, hidden_size)
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

        # query & keys: batch_size * heads, length, hidden_size
        time = T.cast(T.arange(key_length), 'float32')
        td = time[None, :] - time[:, None]
        not_current = T.neq(td, 0)
        after_mask = td > 0
        before_mask = td < 0
        if temporal_bias:
            after = (w_after[:, None, None] *
                     T.switch(after_mask, td, 0)[None, :, :])
            before = (w_before[:, None, None] *
                      T.switch(before_mask, -td, 0)[None, :, :])

        outer_dot = T.batched_tensordot(
            query_hidden, key_hidden, axes=(2, 2)
        ).reshape((
            batch_size, heads, query_length, key_length
        ))

        # outer_dot: batch_size, heads, query_length, key_length
        overall_mask = None
        if generation_mask:
            overall_mask = (
                before_mask[None, None, :, :]
                if overall_mask is None else
                T.and_(overall_mask, before_mask[None, None, :, :])
            )

        if temporal_bias:
            overall_mask = (
                not_current[None, None, :, :]
                if overall_mask is None else
                T.and_(overall_mask, not_current[None, None, :, :])
            )

        if query_mask is not None:
            overall_mask = (
                query_mask[:, None, :, None]
                if overall_mask is None else
                T.and_(overall_mask, query_mask[:, None, :, None])
            )

        if key_mask is not None:
            overall_mask = (
                key_mask[:, None, None, :]
                if overall_mask is None else
                T.and_(overall_mask, key_mask[:, None, None, :])
            )

        attn = softmax(
            (outer_dot / np.float32(np.sqrt(hidden_size))) +
            (after[None, :, :, :] if temporal_bias else 0) +
            (before[None, :, :, :] if temporal_bias else 0) +
            b[None, :, None, None],
            overall_mask,
            axis=3
        )
        tracker.track_variable(P, '%s_attention' % name, attn)
        # attn : batch_size, heads, query_length, key_length
        # values : batch_size, key_length, input_size
        output = T.batched_tensordot(
            attn, value, axes=(3, 1)
        ).dimshuffle(0, 2, 1, 3)
        # output : batch_size, query_length, heads, input_size
        return output
    return attend


def build_layer(P, name, input_size, output_size,
                key_size, hidden_size, heads=1):
    self_attend = build_attention_transform(
        P, name="%s_attn" % name,
        q_size=input_size,
        k_size=input_size,
        hidden_size=key_size,
        heads=heads,
        temporal_bias=True
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
            mask = T.cast(T.ones_like(X[:, :, 0]), 'bool')

        selected = self_attend(
            query=X,
            key=X,
            value=X,
            query_mask=mask,
            key_mask=mask
        )
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
        np.random.randn(2, 5, 10)
    )
    print P.values()
    print transform(X).eval().shape
