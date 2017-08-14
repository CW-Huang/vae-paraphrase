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


def build_step(P, name, input_sizes, hidden_size):
    b = {}
    W = {}
    V = {}
    for gate in ('i', 'f', 'c', 'o'):

        pname = "b_%s_%s" % (name, gate)
        P[pname] = np.zeros((hidden_size,))
        b[gate] = P[pname]

        for i, input_size in enumerate(input_sizes):
            pname = "W_%s_input_%d_%s" % (name, i, gate)
            P[pname] = weight_init(input_size, hidden_size)
            W[i, gate] = P[pname]

        for prev in ('h', 'c'):
            if gate == 'c' and prev == 'c':
                continue
            else:
                pname = "W_%s_%s_%s" % (name, prev, gate)
                P[pname] = transition_init(hidden_size)
                V[prev, gate] = P[pname]

    def _step(*args):
        inputs = args[:len(input_sizes)]
        [prev_cell, prev_hidden] = args[len(input_sizes):]

        i_gate = T.nnet.sigmoid(
            sum(T.dot(x, W[i, 'i']) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, V['h', 'i']) +
            T.dot(prev_cell, V['c', 'i']) +
            b['i']
        )

        f_gate = T.nnet.sigmoid(
            sum(T.dot(x, W[i, 'f']) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, V['h', 'f']) +
            T.dot(prev_cell, V['c', 'f']) +
            b['f']
        )

        cell_ = T.tanh(
            sum(T.dot(x, W[i, 'c']) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, V['h', 'c']) +
            b['c']
        )
        cell = f_gate * prev_cell + i_gate * cell_

        o_gate = T.nnet.sigmoid(
            sum(T.dot(x, W[i, 'o']) for i, x in enumerate(inputs)) +
            T.dot(prev_hidden, V['h', 'o']) +
            T.dot(cell, V['c', 'o']) +
            b['o']
        )

        hidden = o_gate * T.tanh(cell)
        return cell, hidden
    return _step

if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    from pprint import pprint
    P = Parameters()
    step_1 = build_step(
        P, name="test_1",
        input_sizes=[10],
        hidden_size=20
    )
    pprint(P.values())
    output = step_1(
        np.random.randn(1, 10),
        np.random.randn(1, 20),
        np.random.randn(1, 20)
    )
    T.grad(T.sum(output), wrt=P.values())


