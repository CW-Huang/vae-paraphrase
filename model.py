import numpy as np
import theano.tensor as T
import vae
import attn_decoder
# import transformer
import bilstm
import attn_iayn_decoder

def build_annotator(P, hidden_size, embedding_size):
    # layer_count = 3
    # transforms = [None] * layer_count
    # transforms[0] = transformer.build_layer(
    #     P, name="trans_%d" % 0,
    #     input_size=embedding_size,
    #     hidden_size=embedding_size * 2,
    #     output_size=hidden_size,
    #     key_size=32,
    #     heads=4
    # )
    #
    # for i in xrange(1, layer_count):
    #     transforms[i] = transformer.build_layer(
    #         P, name="trans_%d" % i,
    #         input_size=hidden_size,
    #         hidden_size=hidden_size * 2,
    #         output_size=hidden_size,
    #         key_size=32,
    #         heads=4
    #     )
    #
    # def annotate(X, mask):
    #     mask = mask.dimshuffle(1, 0)
    #     prev_layer = X.dimshuffle(1, 0, 2)
    #     for i in xrange(layer_count):
    #         prev_layer = transforms[i](prev_layer, mask)
    #     output = prev_layer.dimshuffle(1, 0, 2)
    #     return output
    process = bilstm.build(
        P, name="encode",
        input_size=embedding_size,
        hidden_size=hidden_size
    )
    return process


def build_self_importance(P, hidden_size):
    P.W_important_picker = 0 * np.random.randn(hidden_size,
                                               hidden_size)

    def select_important(X, mask):
        # X : batch_size, length_size, hidden_size
        # mask : batch_size, length_size
        transformed_X = T.dot(X, P.W_important_picker)
        # transformed_X : batch_size, length_size, hidden_size
        non_diag = T.cast(1 - T.eye(mask.shape[1]), 'int32')
        score = T.switch(
            (mask[:, :, None] and
             mask[:, None, :] and
             non_diag[None, :, :]),
            T.batched_tensordot(
                transformed_X, X,
                axes=(2, 2)
            ), 0
        )
        # score : batch_size, length_size, length_size

        attention = T.nnet.softmax(T.sum(score, axis=-1))
        # attention: batch_size, length_size

        selected = T.batched_dot(attention, X)
        return selected

    return select_important


def build_memory_decoder(P, embedding_size, annotation_size, hidden_size,
                         latent_size, memory_size=5):
    hidden_size = embedding_size
    print embedding_size, latent_size

    _, initial, step = attn_decoder.build(
        P, "memory",
        input_size=hidden_size + latent_size,
        annotation_size=annotation_size,
        hidden_size=hidden_size
    )

    select = build_self_importance(P, hidden_size=annotation_size)

    gaussian_out = vae.build_encoder_output(
        P, name="enc_out",
        input_size=hidden_size,
        output_size=latent_size,
        initialise_weights=lambda x, y: np.random.randn(x, y)
    )

    def decode_memory(annotation_1, annotation_1_mask,
                      annotation_2, annotation_2_mask):
        time_steps, batch_size, annotation_size = annotation_1.shape
        init_vector_1 = select(annotation_2.dimshuffle(1, 0, 2),
                               annotation_2_mask.dimshuffle(1, 0))
        cell_1, hidden_1 = initial(init_vector_1)

        init_vector_2 = select(annotation_1.dimshuffle(1, 0, 2),
                               annotation_1_mask.dimshuffle(1, 0))
        cell_2, hidden_2 = initial(init_vector_2)

        prev_latent = T.zeros((batch_size, latent_size))

        samples_1, means_1, stds_1, samples_2, means_2, stds_2 = \
            [], [], [], [], [], []

        for i in xrange(memory_size):
            inputs_1 = T.concatenate([
                T.zeros((batch_size, embedding_size)),
                prev_latent
            ], axis=1)
            cell_1, hidden_1 = step(inputs_1, cell_1, hidden_1,
                                    annotation_2_mask, annotation_2)
            inputs_2 = T.concatenate([
                hidden_1,
                prev_latent
            ], axis=1)
            cell_2, hidden_2 = step(inputs_2, cell_2, hidden_2,
                                    annotation_1_mask, annotation_1)
            prior_sample, mean_1, std_1 = gaussian_out(hidden_1)
            post_sample, mean_2, std_2 = gaussian_out(hidden_2)

            prev_latent = post_sample
            samples_1.append(prior_sample[None, :, :])
            means_1.append(mean_1[None, :, :])
            stds_1.append(std_1[None, :, :])
            means_2.append(mean_2[None, :, :])
            stds_2.append(std_2[None, :, :])
            samples_2.append(post_sample[None, :, :])

        return (T.concatenate(samples_2, axis=0),
                T.concatenate(means_2, axis=0),
                T.concatenate(stds_2, axis=0),
                T.concatenate(samples_1, axis=0),
                T.concatenate(means_1, axis=0),
                T.concatenate(stds_1, axis=0))

    return decode_memory


def build_encoder(P, embedding_size=256,
                  annotation_size=256,
                  latent_size=64):

    annotate = build_annotator(
        P,
        hidden_size=annotation_size,
        embedding_size=embedding_size,
    )

    decode_memory = build_memory_decoder(
        P,
        embedding_size=embedding_size,
        annotation_size=annotation_size,
        hidden_size=latent_size,
        latent_size=latent_size
    )

    def encode_12(embeddings_12, mask_12, memory_size=5):
        batch_size = embeddings_12.shape[1] // 2

        annotations = annotate(embeddings_12, mask_12)
        annotation_1 = annotations[:, :batch_size, :]
        annotation_2 = annotations[:, batch_size:, :]
        mask_1 = mask_12[:, :batch_size]
        mask_2 = mask_12[:, batch_size:]
        (z_samples, z_means, z_stds,
         _, z_prior_means, z_prior_stds) = \
            decode_memory(annotation_1, mask_1,
                          annotation_2, mask_2)
        return (z_samples,
                z_means, z_stds,
                z_prior_means, z_prior_stds)

    def encode(embeddings_2, mask_2, memory_size=5):
        annotation_2 = annotate(embeddings_2, mask_2)
        mask_2 = T.ones_like(embeddings_2[:, :, 0])
        (_, _, _,
         z_samples, _, _) = \
            decode_memory(annotation_2, mask_2,
                          annotation_2, mask_2)

        return z_samples

    return encode_12, encode


def build_decoder(P, embedding_size,
                  embedding_count,
                  latent_size=64,
                  hidden_size=256):
    decode_ = attn_iayn_decoder.build(
        P, name="decoder",
        embedding_size=embedding_size,
        hidden_size=embedding_size,
        latent_size=latent_size,
        context_size=5
    )
    P.W_output = np.zeros((embedding_size, embedding_count))
    P.b_output = np.zeros((embedding_count,))

    def decode(embeddings_1, mask_1, latent):
        hiddens = decode_(
            embeddings_1,
            latent,
            mask_1
        )
        lin_out = T.dot(hiddens, P.W_output) + P.b_output
        return lin_out

    # def decode_step(x, prev_cell, prev_hidden, latent):
    #     embedding = P.embedding[x]
    #     mask_src = T.ones_like(latent[:, :, 0])
    #     cell, hidden = step(embedding, prev_cell, prev_hidden,
    #                         mask_src, latent)
    #     probs = T.nnet.softmax(T.dot(hidden, P.embedding.T) + P.b_output)
    #     return probs, cell, hidden
    return decode, None, None


def build(P, embedding_size, embedding_count,
          hidden_size=256, latent_size=128):
    P.embedding = np.random.randn(embedding_count,
                                  embedding_size)
    encode, _ = build_encoder(
        P,
        embedding_size=embedding_size,
        annotation_size=hidden_size,
        latent_size=latent_size
    )

    decode, _, _ = build_decoder(
        P,
        embedding_size=embedding_size,
        embedding_count=embedding_count,
        latent_size=latent_size,
        hidden_size=hidden_size
    )

    def cost(X_12):
        batch_size = X_12.shape[0] // 2
        X_12 = X_12.T
        embeddings = P.embedding[X_12]
        mask = T.neq(X_12, -1)
        (z_samples,
         z_means, z_stds,
         z_prior_means, z_prior_stds) = encode(embeddings, mask, memory_size=5)
        embeddings_1 = embeddings[:, :batch_size, :]
        mask_1 = mask[:, :batch_size]
        X_1 = X_12[:, :batch_size]
        lin_output = decode(embeddings_1[:-1], mask_1[:-1], z_samples)
        kl = T.sum(vae.kl_divergence(
                z_means, z_stds,
                z_prior_means, z_prior_stds
            ), axis=(0, 1)
        )
        recon = T.sum(recon_cost(lin_output, X_1[1:]))
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
    P.embedding = np.random.randn(5, 20)

    X_12 = T.as_tensor_variable(
        np.array([
            [  0,  1,  2, -1, -1, -1],
            [  0,  1,  2,  3,  4, -1],
            [  0,  1,  2,  3, -1, -1],
            [  0,  1,  2,  3,  4,  1]
        ]).astype(np.int32)
    )
    encoder, _ = build_encoder(P, embedding_size=20,
                               annotation_size=20,
                               latent_size=16)
    val = encoder(P.embedding[X_12.T], T.neq(X_12.T, -1))[0].eval()
    print val.shape

