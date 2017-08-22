import lstm
import numpy as np
import theano
import theano.tensor as T
import feedforward


def build(P, name, input_size, hidden_size):

    lstm_step_f, non_sequences_f = lstm.build_step(
        P, name="%s_forward" % name,
        input_sizes=[input_size],
        hidden_size=hidden_size,
    )

    lstm_step_b, non_sequences_b = lstm.build_step(
        P, name="%s_backward" % name,
        input_sizes=[input_size],
        hidden_size=hidden_size,
    )

    P['init_%s_forward_hidden' % name] = np.zeros((hidden_size,))
    P['init_%s_forward_cell' % name] = np.zeros((hidden_size,))
    P['init_%s_backward_hidden' % name] = np.zeros((hidden_size,))
    P['init_%s_backward_cell' % name] = np.zeros((hidden_size,))

    init_forward_hidden = P['init_%s_forward_hidden' % name]
    init_forward_cell = P['init_%s_forward_cell' % name]
    init_backward_hidden = P['init_%s_backward_hidden' % name]
    init_backward_cell = P['init_%s_backward_cell' % name]

    output_transform = feedforward.build_combine_transform(
        P, name=name,
        input_sizes=[hidden_size] * 2,
        output_size=hidden_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu
    )

    def _step(mask_f, mask_b,
              embedding_f, embedding_b,
              prev_cell_f, prev_hidden_f,
              prev_cell_b, prev_hidden_b, *non_sequences):
        f_non_seq = non_sequences[:len(non_sequences_f)]
        b_non_seq = non_sequences[len(non_sequences_f):]
        cell_f, hidden_f = lstm_step_f(embedding_f, prev_cell_f, prev_hidden_f,
                                       *f_non_seq)
        cell_b, hidden_b = lstm_step_b(embedding_b, prev_cell_b, prev_hidden_b,
                                       *b_non_seq)
        cell_f = cell_f  # T.switch(mask_f, cell_f, prev_cell_f)
        cell_b = T.switch(mask_b, cell_b, prev_cell_b)
        hidden_f = hidden_f  # T.switch(mask_f, hidden_f, prev_hidden_f)
        hidden_b = T.switch(mask_b, hidden_b, prev_hidden_b)
        return cell_f, hidden_f, cell_b, hidden_b

    def initial(batch_size):
        init_forward_hidden_batch = T.alloc(T.tanh(init_forward_hidden), batch_size, hidden_size)
        init_forward_cell_batch = T.alloc(init_forward_cell, batch_size, hidden_size)
        init_backward_hidden_batch = T.alloc(T.tanh(init_backward_hidden), batch_size, hidden_size)
        init_backward_cell_batch = T.alloc(init_backward_cell, batch_size, hidden_size)
        return (init_forward_hidden_batch,
                init_forward_cell_batch,
                init_backward_hidden_batch,
                init_backward_cell_batch)

    def process(X, mask):
        mask_f = mask[:, :, None]
        mask_b = mask_f[::-1]
        X_f = X
        X_b = X[::-1]
        [cells_f, hiddens_f, cells_b, hiddens_b], _ = theano.scan(
            _step,
            sequences=[mask_f, mask_b, X_f, X_b],
            outputs_info=initial(X.shape[1]),
            non_sequences=non_sequences_f + non_sequences_b,
            strict=True
        )
        outputs = output_transform([hiddens_f, hiddens_b[::-1]])
        outputs = T.switch(mask_f, outputs, 0)
        return outputs
    return process
