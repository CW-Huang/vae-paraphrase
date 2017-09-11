import numpy as np
import theano.tensor as T
import vae
import attn_decoder
import transformer
#import bilstm
import attn_iayn_decoder


def build_annotator(P, hidden_size, embedding_size):
    layer_count = 4
    transforms = [None] * layer_count
    transforms[0] = transformer.build_layer(
        P, name="trans_%d" % 0,
        input_size=embedding_size,
        hidden_size=hidden_size,
        output_size=hidden_size,
        key_size=64,
        heads=4
    )

    for i in xrange(1, layer_count):
        transforms[i] = transformer.build_layer(
            P, name="trans_%d" % i,
            input_size=hidden_size,
            hidden_size=hidden_size * 2,
            output_size=hidden_size,
            key_size=64,
            heads=4
        )

    def process(X, mask):
        mask = mask.dimshuffle(1, 0)
        prev_layer = X.dimshuffle(1, 0, 2)
        for i in xrange(layer_count):
            prev_layer = transforms[i](prev_layer, mask)
        output = prev_layer.dimshuffle(1, 0, 2)
        return output
        
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
        initialise_weights=lambda x, y: 0. * np.random.randn(x, y)
    )

    def decode_memory(annotation, annotation_mask,
                      batch_size,num_paraphrases):
                          
                          
        time_steps, n, annotation_size = annotation.shape
        init_vector = select(annotation.dimshuffle(1, 0, 2),
                             annotation_mask.dimshuffle(1, 0))
        cells, hiddens = initial(init_vector)


        prev_latent = T.zeros((n, latent_size))

        samples_2, means_2, stds_2 = [], [], []
        
        for i in xrange(memory_size):
            inputs_1 = T.concatenate([
                T.zeros((n, embedding_size)),
                prev_latent
            ], axis=1)
            cells, hiddens = step(inputs_1, cells, hiddens,
                                  annotation_mask, annotation)
            
            
            post_sample, mean_2, std_2 = gaussian_out(
                hiddens.reshape((batch_size,num_paraphrases,-1)).mean(1)
            )
            
            #post_sample, mean_2, std_2 = gaussian_out(hidden_2)

            #prev_latent = post_sample # TODO: auto-regressive posterior


            #samples_1.append(prior_sample[None, :, :])
            #means_1.append(mean_1[None, :, :])
            #stds_1.append(std_1[None, :, :])
            means_2.append(mean_2[None, :, :])
            stds_2.append(std_2[None, :, :])
            samples_2.append(post_sample[None, :, :])

        return (T.concatenate(samples_2, axis=0),
                T.concatenate(means_2, axis=0),
                T.concatenate(stds_2, axis=0),
                #T.concatenate(samples_1, axis=0),
                #T.concatenate(means_1, axis=0),
                #T.concatenate(stds_1, axis=0))
                )
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
        # X_12 : bs, n_paras, length, dim_embeddings
        batch_size = embeddings_12.shape[0]
        n_paraphrases = embeddings_12.shape[1]
        length = embeddings_12.shape[2]
        n = batch_size * n_paraphrases 

        
        annotations = annotate(embeddings_12.reshape((n,length,-1)).dimshuffle(1,0,2), 
                               mask_12.reshape((n,length)).T)
        # annotations: time_steps, n, annotation_size
        #annotation_1 = annotations[:, :batch_size, :]
        #annotation_2 = annotations[:, batch_size:, :]
        #mask_1 = mask_12[:, :batch_size]
        #mask_2 = mask_12[:, batch_size:]
                               
        (z_samples, z_means, z_stds) = \
            decode_memory(annotations, mask_12.reshape((n,length)).T,
                          batch_size, n_paraphrases)

       
        return (z_samples,
                z_means, z_stds)

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
    P.b_output = np.zeros((embedding_count,))

    def decode(embeddings_1, mask_1, latent):
        hiddens = decode_(
            embeddings_1,
            latent,
            mask_1
        )
        lin_out = T.dot(hiddens, P.embedding.T) + P.b_output
        return lin_out

    def decode_step(x, latent):
        # x : accumulated_idxs x batch_size
        embeddings = P.embedding[x]
        hidden = decode_(embeddings, latent,
                         T.ones_like(x))[-1]
        probs = T.nnet.softmax(
            T.dot(hidden, P.embedding.T) + P.b_output
        )
        return probs
    return decode, decode_step


def build(P, embedding_size, embedding_count,
          hidden_size=256, latent_size=128, memory_size=5):
    P.embedding = np.random.randn(embedding_count,
                                  embedding_size)
    encode, _ = build_encoder(
        P,
        embedding_size=embedding_size,
        annotation_size=hidden_size,
        latent_size=latent_size
    )

    decode, _ = build_decoder(
        P,
        embedding_size=embedding_size,
        embedding_count=embedding_count,
        latent_size=latent_size,
        hidden_size=hidden_size
    )

    def cost(X_12):
        # X_12 : bs, n_paras, length
    
        batch_size = X_12.shape[0]
        num_paras = X_12.shape[1]
        length = X_12.shape[2]
        n = batch_size * num_paras
        
        #X_12 = X_12.T # TODO: correct the following part
        
        embeddings = P.embedding[X_12]
        mask = T.neq(X_12, -1)
        (z_samples,
         z_means, z_stds) = encode(embeddings, mask, memory_size=memory_size)
        
        #embeddings_1 = embeddings[:, :batch_size, :]
        #mask_1 = mask[:, :batch_size]
        X_1 = X_12[:, :batch_size]
        
        # bs, n_paras, mem, latent
        ones_ = T.ones((batch_size,num_paras,memory_size,latent_size))  
        def process_(var):
            var_o = var.dimshuffle(1,'x',0,2)  * ones_
            var_r = var_o.reshape((-1,memory_size,latent_size))
            return var_r.dimshuffle(1,0,2)
            
        z_samples_shared = process_(z_samples)
        z_means_shared = process_(z_means)
        z_stds_shared = process_(z_stds)
        # mem, n, latent
        
        
        embs_dec = embeddings.reshape((n,length,-1)).dimshuffle(1,0,2)[:-1]
        mask_dec = mask.reshape((n,length)).dimshuffle(1,0)[:-1]

        lin_output = decode(embs_dec, mask_dec, z_samples_shared)
        
        kl = T.sum(vae.kl_divergence(
                z_means_shared, z_stds_shared,
                0, 1 # TODO: std gaussian prior
            ), axis=(0, 1)
        )
        
        target = X_1.reshape((n,length)).dimshuffle(1,0)[1:]
        recon = T.sum(recon_cost(lin_output, target))
        return recon, kl, target, lin_output, z_means_shared, z_stds_shared, z_stds
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
    
#    
#    # test 1
#    P.embedding = np.random.randn(5, 20)
#    X_12 = T.as_tensor_variable(
#        np.array([
#            [  0,  1,  2, -1, -1, -1],
#            [  0,  1,  2,  3,  4, -1],
#            [  0,  1,  2,  3, -1, -1],
#            [  0,  1,  2,  3,  4,  1]
#        ]).astype(np.int32).reshape(4,1,6)
#    )
#    encoder, _ = build_encoder(P, embedding_size=20,
#                               annotation_size=20,
#                               latent_size=16)
#    encs = encoder(P.embedding[X_12], T.neq(X_12, -1))
#    val = encs[0].eval()
#    print val.shape
#    encs[0].eval()
    
    # test 2
    X_12 = T.as_tensor_variable(
        np.array([
            [  0,  1,  2, -1, -1, -1],
            [  0,  1,  2,  3,  4, -1],
            [  0,  1,  2,  3, -1, -1],
            [  0,  1,  2,  3,  4,  1]
        ]).astype(np.int32).reshape(4,1,6)
    )
    
    
    cost = build(
        P,
        embedding_count=5,
        embedding_size=20,
        hidden_size = 128
    )
    loss = cost(X_12)
    
    print loss[0].eval()
    print loss[1].eval()
                          
                          
                          