import numpy as np
import theano.tensor as T
import vae
import bilstm
import attn_decoder


def build_bilstm(P, hidden_size, embedding_size):
    process = bilstm.build(
        P, name="encode",
        input_size=embedding_size,
        hidden_size=hidden_size
    )
    return process


def build_memory_decoder(P, embedding_size, annotation_size, hidden_size):
    mem_decode, _, _ = attn_decoder.build(
        P, "memory",
        embedding_size,
        annotation_size,
        hidden_size=embedding_size
    )

    def decode_memory(ann_mask, basis_mask, annotation, basis):
        return mem_decode(ann_mask, basis_mask, annotation, basis)
    return decode_memory


def build_encoder(P, embedding_size=256,
                  annotation_size=256,
                  latent_size=64):

    annotate = build_bilstm(
        P,
        hidden_size=annotation_size,
        embedding_size=embedding_size,
    )

    decode_memory = build_memory_decoder(
        P,
        embedding_size=embedding_size,
        annotation_size=annotation_size,
        hidden_size=latent_size
    )

    gaussian_out = vae.build_encoder_output(
        P, name="enc_out",
        input_size=annotation_size,
        output_size=latent_size
    )

    def encode_12(embeddings_12, mask_12, memory_size=5):
        batch_size = embeddings_12.shape[1] // 2

        annotations = annotate(embeddings_12, mask_12)
        annotation_1 = annotations[:, :batch_size, :]
        annotation_2 = annotations[:, batch_size:, :]

        mask_1 = mask_12[:, :batch_size]
        mask_2 = mask_12[:, batch_size:]
        mask_dst = T.ones((memory_size, batch_size))

        memory_0 = T.zeros((memory_size, batch_size, embedding_size))
        memory_1 = decode_memory(mask_dst, mask_2, memory_0, annotation_2)
        memory_2 = decode_memory(mask_dst, mask_1, memory_1, annotation_1)

        (_, z_prior_means, z_prior_stds) = gaussian_out(memory_1)
        (z_samples, z_means, z_stds) = gaussian_out(memory_2)

        return (z_samples,
                z_means, z_stds,
                z_prior_means, z_prior_stds)

    def encode(embeddings_2, mask_2, memory_size=5):
        batch_size = embeddings_2.shape[1]
        annotation_2 = annotate(embeddings_2, mask_2)
        mask_dst = T.ones((memory_size, batch_size))

        memory_0 = T.zeros((memory_size, batch_size, embedding_size))
        memory_1 = decode_memory(mask_dst, mask_2, memory_0, annotation_2)
        (z_samples, _, _) = gaussian_out(memory_1)
        return z_samples

    return encode_12, encode


def build_decoder(P, embedding_size,
                  embedding_count,
                  latent_size=64,
                  hidden_size=256):
    decode_, initial, step = attn_decoder.build(
        P, name="decoder",
        embedding_size=embedding_size,
        annotation_size=latent_size,
        hidden_size=hidden_size
    )
    P.b_output = np.zeros((embedding_count,))

    def decode(embeddings_1, mask_1, latent):
        hiddens = decode_(
            mask_dst=mask_1,
            mask_src=T.ones_like(latent[:, :, 0]),
            embeddings=embeddings_1,
            annotation=latent
        )
        lin_out = T.dot(hiddens, P.embedding.T) + P.b_output
        return lin_out

    def decode_step(x, prev_cell, prev_hidden, latent):
        embedding = P.embedding[x]
        mask_src = T.ones_like(latent[:, :, 0])
        return step(embedding, prev_cell, prev_hidden, mask_src, latent)
    return decode, initial, decode_step


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
#            [  0,  1,  2,  3,  4, -1],
#            [  0,  1,  2,  3, -1, -1],
#            [  0,  1,  2,  3,  4,  1]
        ]).astype(np.int32)
    )
    _, encoder = build_encoder(P, embedding_size=20,
                               annotation_size=20,
                               latent_size=16)
    val = encoder(P.embedding[X_12.T], T.neq(X_12.T, -1)).eval()
    print val

