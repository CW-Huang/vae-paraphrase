import numpy as np
import theano.tensor as T
import vae
import theano_toolkit.utils as U
import gated_seq_conv
import attn_decoder


def build_encoder(P, embedding_size, latent_size):
    encode_ = gated_seq_conv.build(
        P, name="encoder",
        input_size=embedding_size,
        window_size=3
    )

    gaussian_output = vae.build_encoder_output(
        P, name="encoder_gaussian",
        input_size=embedding_size,
        output_size=latent_size,
        initialise_weights=None
    )

    def encode(X, mask):
        mask = mask[:, :, None]
        hiddens, _ = encode_(X, mask)
        output = gaussian_output(hiddens)
        _, means, stds = output
        means = T.switch(mask, means, 0)
        stds = T.switch(mask, stds, 1)
        return means, stds
    return encode


def build_decoder(P, embedding_size, latent_size):
    decode_, _, _ = attn_decoder.build_decoder(
        P,
        embedding_size=embedding_size,
        latent_size=latent_size
    )
    P.b_output = np.zeros((P.embedding.get_value().shape[0],))

    def decode(mask, embeddings, latent):
        mask_src = mask
        mask_dst = mask[:, :-1]
        hiddens = decode_(mask_dst, mask_src, embeddings, latent)
        return T.dot(hiddens, P.embedding.T) + P.b_output
    return decode


def build(P, embedding_size, embedding_count,
          hidden_size=128, latent_size=128):
    P.embedding = np.random.randn(embedding_count,
                                  embedding_size)

    encode = build_encoder(P, embedding_size, latent_size)
    decode = build_decoder(P, embedding_size, latent_size)

    def encode(embeddings, mask):
        means, stds = encode(embeddings, mask)
        eps = U.theano_rng.normal(size=stds.shape)
        samples = means + eps * stds
        return samples, means, stds

    def encode_decode(X_12):
        batch_size = X_12.shape[0] // 2

        embeddings = P.embedding[X_12]
        mask = T.neq(X_12, -1)
        _, means, stds = encode(X_12, mask)

        mask_1 = mask[:batch_size]
        embeddings_1 = embeddings[:batch_size]

        Z_1_means, Z_1_stds = means[:batch_size], stds[:batch_size]
        Z_2_means, Z_2_stds = means[batch_size:], stds[batch_size:]

        eps = U.theano_rng.normal(size=Z_1_stds.shape)
        Z_1 = Z_1_means + eps * Z_1_stds
        lin_output = decode(mask_1, embeddings_1, Z_1)
        return (lin_output,
                Z_1_means, Z_1_stds,
                Z_2_means, Z_2_stds)

    def cost(X_12):
        batch_size = X_12.shape[0] // 2
        X_1 = X_12[:batch_size]
        (lin_output,
         Z_1_means, Z_1_stds,
         Z_2_means, Z_2_stds) = encode_decode(X_12)
        kl = T.sum(vae.kl_divergence(
                Z_1_means, Z_1_stds,
                Z_2_means, Z_2_stds
            ), axis=(0, 1))
        recon = T.sum(recon_cost(lin_output, X_1[:, 1:].T))
        return recon, kl
    return cost


def recon_cost(output_lin, labels):
    output = T.nnet.softmax(
        output_lin.reshape((
            output_lin.shape[0] * output_lin.shape[1],
            output_lin.shape[2]
        ))
    )

    labels = labels.reshape((labels.shape[0] * labels.shape[1],))
    mask = T.neq(labels, -1)
    labels = T.switch(mask, labels, 0)
    labels = T.cast(labels, 'int32')

    xent = T.switch(mask,
                    T.nnet.categorical_crossentropy(output, labels), 0)
    return xent


if __name__ == "__main__":
    from theano_toolkit.parameters import Parameters
    P = Parameters()
    X_12 = T.as_tensor_variable(
        np.array([
            [  0,  1,  2, -1, -1, -1],
            [  0,  1,  2,  3,  4, -1],
            [  0,  1,  2,  3, -1, -1],
            [  0,  1,  2,  3,  4,  1]
        ]).astype(np.int32)
    )
    cost = build(
        P, embedding_size=64, embedding_count=10,
        hidden_size=128, latent_size=128
    )
    recon, kl = cost(X_12)
