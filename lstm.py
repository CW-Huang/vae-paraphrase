import theano
import theano.tensor as T
import numpy as np
import scipy.linalg


def weight_init(input_size, output_size, factor=1):
    return np.asarray(
      np.random.uniform(
         low=-factor * np.sqrt(6. / (input_size + output_size)),
         high=factor * np.sqrt(6. / (input_size + output_size)),
         size=(input_size, output_size)
      ),
      dtype=theano.config.floatX
    )


def transition_init(size):
    return scipy.linalg.orth(np.random.randn(size, size))


def build(P, name, input_size, hidden_size, truncate_gradient=-1):
    name_init_hidden = "init_%s_hidden" % name
    name_init_cell = "init_%s_cell" % name
    P[name_init_hidden] = np.zeros((hidden_size,))
    P[name_init_cell] = np.zeros((hidden_size,))

    _step = build_step(P, name, [input_size], hidden_size)

    def lstm_layer(X):
        init_hidden = T.tanh(P[name_init_hidden])
        init_cell = P[name_init_cell]
        init_hidden_batch = T.alloc(init_hidden, X.shape[1], hidden_size)
        init_cell_batch = T.alloc(init_cell, X.shape[1], hidden_size)
        [cell, hidden], _ = theano.scan(
            _step,
            sequences=[X],
            outputs_info=[init_cell_batch, init_hidden_batch],
            truncate_gradient=truncate_gradient
        )
        return cell, hidden
    return lstm_layer


def build_step(P, name, input_sizes, hidden_size, strict=False):
    input_count = len(input_sizes)
    W = []
    V = []
    b = []

    for gate in ('i', 'f', 'c', 'o'):
        for i, input_size in enumerate(input_sizes):
            pname = "W_%s_input_%d_%s" % (name, i, gate)
            P[pname] = weight_init(input_size, hidden_size)
            W.append(P[pname])

        for prev in ('h', 'c'):
            if gate == 'c' and prev == 'c':
                continue
            else:
                pname = "W_%s_%s_%s" % (name, prev, gate)
                P[pname] = transition_init(hidden_size)
                V.append(P[pname])

        pname = "b_%s_%s" % (name, gate)
        P[pname] = np.zeros((hidden_size,))
        b.append(P[pname])

    assert(len(W) == input_count * 4)
    assert(len(V) == 7)
    assert(len(b) == 4)

    def _step(*args):
        inputs, args = args[:input_count], args[input_count:]

        [prev_cell, prev_hidden], args = args[:2], args[2:]

        W_x_i, args = args[:input_count], args[input_count:]
        W_x_f, args = args[:input_count], args[input_count:]
        W_x_c, args = args[:input_count], args[input_count:]
        W_x_o, args = args[:input_count], args[input_count:]
        (W_h_i, W_c_i,
         W_h_f, W_c_f,
         W_h_c,
         W_h_o, W_c_o), args = args[:7], args[7:]
        (b_i, b_f, b_c, b_o) = args
        i_gate = T.nnet.sigmoid(
            sum(T.dot(x, W_x_i[i]) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, W_h_i) + T.dot(prev_cell, W_c_i) +
            b_i
        )

        f_gate = T.nnet.sigmoid(
            sum(T.dot(x, W_x_f[i]) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, W_h_f) +
            T.dot(prev_cell, W_c_f) +
            b_f
        )

        cell_ = T.tanh(
            sum(T.dot(x, W_x_c[i]) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, W_h_c) +
            b_c
        )
        cell = f_gate * prev_cell + i_gate * cell_

        o_gate = T.nnet.sigmoid(
            sum(T.dot(x, W_x_o[i]) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, W_h_o) + T.dot(cell, W_c_o) +
            b_o
        )

        hidden = o_gate * T.tanh(cell)
        return cell, hidden
    return _step, W + V + b

if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    from pprint import pprint
    P = Parameters()
    step_1, non_sequences = build_step(
        P, name="test_1",
        input_sizes=[10],
        hidden_size=20
    )
    pprint(non_sequences)
    pprint(P.values())
    output = step_1(
        np.random.randn(1, 10),
        np.random.randn(1, 20),
        np.random.randn(1, 20)
    )
    T.grad(T.sum(output), wrt=P.values())
