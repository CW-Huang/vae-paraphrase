import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U

import feedforward


def build(P, name,
          input_size,
          encoder_hidden_sizes,
          latent_size,
          decoder_hidden_sizes=None,
          activation=T.nnet.softplus,
          initial_weights=feedforward.relu_init):

    if decoder_hidden_sizes is None:
        decoder_hidden_sizes = encoder_hidden_sizes[::-1]

    encode = build_inferer(P, "%s_encoder" % name,
                           [input_size],
                           encoder_hidden_sizes,
                           latent_size,
                           activation=activation,
                           initial_weights=initial_weights,
                           initialise_outputs=True
                           )
    decode = build_inferer(P, "%s_decoder" % name,
                           [latent_size],
                           decoder_hidden_sizes,
                           input_size,
                           activation=activation,
                           initial_weights=initial_weights)
    return encode, decode


def gaussian_nll(X, mean, std):
    return np.float32(0.5) * T.sum(
            np.log(2 * np.pi) + 2 * T.log(std) +
            T.sqr(X - mean) / T.sqr(std), axis=-1
        )


def kl_divergence(mean, std, prior_mean, prior_std):

    output = np.float32(0.5) * T.sum(
            np.float32(2) * T.log(prior_std) - np.float32(2) * T.log(std) +
            ((T.sqr(std) + T.sqr(mean - prior_mean)) /
                T.sqr(prior_std)) - np.float32(1), axis=-1
        )
    return output


def build_inferer(P, name, input_sizes, hidden_sizes, output_size,
                  initial_weights, activation,
                  initialise_outputs=False):

    combine_inputs = feedforward.build_combine_transform(
        P, "%s_input" % name,
        input_sizes, hidden_sizes[0],
        initial_weights=initial_weights,
        activation=activation,
    )

    transform = feedforward.build_stacked_transforms(
        P, name, hidden_sizes,
        initial_weights=initial_weights,
        activation=activation)

    output = build_encoder_output(
        P, name,
        hidden_sizes[-1], output_size,
        initialise_weights=(initial_weights if initialise_outputs else None)
    )

    def infer(Xs, samples=-1):
        combine = combine_inputs(Xs)
        hiddens = transform(combine)
        latent, mean, std = output(hiddens[-1], samples=samples)
        return latent, mean, std
    return infer


def build_encoder_output(P, name, input_size, output_size,
                         initialise_weights=None):

    if initialise_weights is None:
        def initialise_weights(x, y):
            return np.zeros((x, y))

    P["W_%s_mean" % name] = initialise_weights(input_size, output_size)
    P["b_%s_mean" % name] = np.zeros((output_size,))
    P["W_%s_std" % name] = np.zeros((input_size, output_size))
    P["b_%s_std" % name] = np.zeros((output_size,)) + np.log(np.exp(1) - 1)

    def output(X, samples=-1):
        mean = T.dot(X, P["W_%s_mean" % name]) + P["b_%s_mean" % name]
        std = T.nnet.softplus(T.dot(X, P["W_%s_std" % name]) +
                              P["b_%s_std" % name]) + 1e-5
        eps = U.theano_rng.normal(size=std.shape)
        latent = mean + eps * std
        return latent, mean, std
    return output
