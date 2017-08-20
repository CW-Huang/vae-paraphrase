import theano
import theano.tensor as T
import numpy as np
import scipy.linalg

import theano_toolkit.utils as U


def weight_init(input_size, output_size, window_size):
    factor = 1
    input_size_ = window_size * input_size
    return np.asarray(
      np.random.uniform(
         low=-factor * np.sqrt(6. / (input_size_ + output_size)),
         high=factor * np.sqrt(6. / (input_size_ + output_size)),
         size=(output_size, input_size, 1, window_size)
      ),
      dtype=theano.config.floatX
    )



def transition_init(size):
    return scipy.linalg.orth(np.random.randn(size, size))


def build_1d_conv(P, name, input_size, output_size, window_size,
                  activation=T.tanh,
                  weight_init=weight_init):
    P["W_%s" % name] = weight_init(input_size, output_size, window_size)
    b = np.zeros((output_size,))
    P["b_%s" % name] = b
    W = P["W_%s" % name]
    b = P["b_%s" % name].dimshuffle('x', 0, 'x', 'x')
    def conv(x, W=W, b=b, shuffle_dim=True):
        if shuffle_dim:
            x = x.dimshuffle(0, 2, 'x', 1)
        # batch_size x hidden_size x 1 x sequence_length
        conv_out_ = T.nnet.conv2d(x, W, border_mode='half') + b
        output = activation(conv_out_)
        if shuffle_dim:
            output = output.dimshuffle(2, 0, 3, 1)[0]
        return output
    return conv, [W, b]



def build_op(P, name, input_size, window_size):
    conv_, params = build_1d_conv(
        P, name="gated_%s" % name,
        input_size=input_size,
        output_size=input_size + 1,
        window_size=window_size,
        activation=lambda x: x,
        weight_init=weight_init
    )
    W, b = params
    def conv(x, W=W, b=b, shuffle_dim=True):
        if shuffle_dim:
            x = x.dimshuffle(0, 2, 'x', 1)
        # batch_size x hidden_size x 1 x sequence_length
        output_ = conv_(x, shuffle_dim=False)
        output = T.tanh(output_[:, :-1])
        g = T.nnet.sigmoid(output_[:, -1])[:, None, :, :]

        output = g * output + (1 - g) * x

        if shuffle_dim:
            output = output.dimshuffle(2, 0, 3, 1)[0]
        return output, g
    return conv, params


def build(P, name, input_size, window_size):
    gated_conv, params = build_op(
        P, name=name,
        input_size=input_size,
        window_size=window_size,
    )

    def _step(prev_state, mask, W, b):
        prev_state, gate = gated_conv(prev_state, W, b, shuffle_dim=False)
        return T.switch(mask, prev_state, 0), gate


    def process(X, mask, times=None, training=True):
        if times is None:
            times = X.shape[-2]
        X = T.patternbroadcast(
            X.dimshuffle(0, 2, 'x', 1),
            (False, False, False, False)
        )
        mask = mask.dimshuffle(0, 'x', 'x', 1)
        [outputs, gates], _ = theano.scan(
            _step,
            outputs_info=[X, None],
            non_sequences=[mask] + params,
            n_steps=times,
            strict=True
        )
        # outputs : time x batch_size x hidden_size x 1 x length
        output = outputs[-1, :, :, 0].dimshuffle(0, 2, 1)
        return output, gates
    return process



if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    from theano_toolkit import hinton
    P = Parameters()

    conv = build(
        P, name="test",
        input_size=5,
        window_size=3
    )

    test_input = T.as_tensor_variable(
        np.random.randn(1, 20, 5).astype(np.float32)
    )
    output, gates = conv(test_input, mask=T.ones_like(test_input[:, :, 0]))
    val = gates.eval()
    hinton.plot(val)
    print val.shape
    print val




