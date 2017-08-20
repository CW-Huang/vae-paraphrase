import theano
import theano.tensor as T
import numpy as np
import scipy.linalg

import theano_toolkit.utils as U


def weight_init(input_size, output_size, window_size):
    factor = np.sqrt(6. / (input_size * window_size + output_size))
    return np.asarray(
      np.random.uniform(
         low=-factor, high=factor,
         size=(output_size, input_size, 1, window_size)
      ),
      dtype=theano.config.floatX
    )


def transition_init(size):
    return scipy.linalg.orth(np.random.randn(size, size))


def build(P, name, input_size, output_size, window_size):
    P["W_%s"] = weight_init(input_size, output_size + 1, window_size)
    b = np.zeros((output_size + 1,))
    b[:-1] = 5
    P["b_%s"] = b
    W = P["W_%s"]
    b = P["b_%s"].dimshuffle('x', 0, 'x', 'x')
    def conv(x, shuffle_dim=True):
        if shuffle_dim:
            x = x.dimshuffle(0, 2, 'x', 1)
        # batch_size x hidden_size x 1 x sequence_length
        conv_out_ = T.nnet.conv2d(x, W, border_mode='valid') + b
        conv_out = T.concatenate([
            conv_out_,
            conv_out_[:, :, :, -1:]
        ], axis=3)
        merge_gate = T.nnet.sigmoid(conv_out[:, -1, :, :]).dimshuffle(0, 'x', 1, 2)
        # batch_size x 1 x 1 x sequence_length
        conv_out = (merge_gate * conv_out[:, :-1, :, :] +
                    (1 - merge_gate) * x)
        merge_gate = merge_gate[:, 0, 0, :]
        #merge_gate = U.theano_rng.uniform(size=merge_gate.shape) > merge_gate
        #merge_gate = 0.005 + (1 - 0.01) * merge_gate

        exist_gate = calculate_existence_gates(merge_gate)
        exist_gate = exist_gate[:, None, None, :]

        output = (
            (exist_gate * conv_out)[:, :, :, :-1] +
            ((1 - exist_gate) * conv_out)[:, :, :, 1:]
        )

        if shuffle_dim:
            output = output.dimshuffle(2, 0, 3, 1)[0]
        return output, merge_gate, exist_gate
    return conv

def calculate_existence_gates(g):
    idx = T.arange(g.shape[1])
    top_right_mat = idx[:, None] < idx[None, :]
    top_right_mat_ = idx[:, None] <= idx[None, :]
    alternate_sign = (-1) ** idx
    signs = alternate_sign[None, :] * alternate_sign[:, None]

    log_g = T.log(g)
    log_cumul_g = T.dot(log_g, top_right_mat)
    combinations = T.exp(
        log_cumul_g[:, None, :] -
        log_cumul_g[:, :, None]
    )

    combinations = T.switch(top_right_mat_, signs * combinations, 0)
    exist_gate = T.sum(combinations, axis=1)
    exist_gate = T.set_subtensor(exist_gate[:, 0], 1)
    return exist_gate

if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    from theano_toolkit import hinton
    P = Parameters()

    conv = build(
        P, name="test",
        input_size=5,
        output_size=5,
        window_size=2
    )

    test_input = T.as_tensor_variable(
        np.random.randn(1, 20, 5).astype(np.float32)
    )
    output, merge_gate, exist_gate = conv(test_input)
    val = output.eval()
    hinton.plot(val)
    print val.shape
    print val




