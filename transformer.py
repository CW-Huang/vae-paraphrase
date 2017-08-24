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

def build_self_attention(P, name, input_size, key_size):
    P['W_%s_query' % name] = np.random.randn(input_size, key_size)
    P['W_%s_key' % name] = np.random.randn(input_size, key_size)
    P['w_%s_after' % name] = np.float32(0)
    P['w_%s_before' % name] = np.float32(0)
    P['b_%s' % name] = np.float32(0)
    W_query = P['W_%s_query' % name]
    W_key = P['W_%s_key' % name]
    w_after = P['w_%s_after' % name]
    w_before = P['w_%s_before' % name]
    b = P['b_%s' % name]
    def self_attend(values, mask=None):
        if mask is None:
            mask = T.ones_like(values[:, :, 0])
        mask = mask[:, :, None]
        query = T.dot(values, W_query)
        keys = T.dot(values, W_key)
        time = T.arange(values.shape[1])
        td = time[None, :] - time[:, None]
        current = T.neq(td, 0)
        after = w_after * T.switch(td > 0, td, 0)
        before = w_before * T.switch(td < 0, -td, 0)
        # query : batch_size x length x key_size
        # keys : batch_size x length x key_size
        # values : batch_size x length x value_size
        attn = softmax(
            (T.batched_tensordot(
                 query, keys,
                 axes=(2, 2)
             ) / np.sqrt(key_size) +
             after + before + b),
            mask and current,
            axis=-1
        )
        from theano_toolkit import hinton
        hinton.plot(attn.eval())

        # attn : batch_size x length x length
        output = T.batched_dot(attn, values)
        return output
    return self_attend

def build_transform_layer(P, name, input_size, output_size,
                          key_size, hidden_size):
    self_attend = build_self_attention(
        P, name="%s_attn" % name,
        input_size=input_size,
        key_size=key_size
    )

    layer_transform = feedforward.build_classifier(
        P, name="%s_transform" % name,
        input_sizes=[input_size, input_size],
        hidden_sizes=[hidden_size],
        output_size=output_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        output_activation=lambda x: x,
        output_weights=feedforward.relu_init
    )

    def transform(X):
        selected = self_attend(X)
        _, output = layer_transform([X, selected])
        return output
    return transform


if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    transform = build_transform_layer(
        P, name="test",
        input_size=20,
        output_size=30,
        key_size=20,
        hidden_size=20
    )
    X = T.as_tensor_variable(
        np.random.randn(1, 10, 20)
    )
    print P.values()
    print transform(X).eval().shape

