import pandas as pd
import numpy as np
import collections

batch_size = 512
embedding_size = 100 # Dimension of the embedding vector.
num_sampled = 10 # Number of negative examples to sample.
vocab_size = 40000
window_size = 5
context_window_size = 2*window_size
num_negative_sample = 5

class ParagraphVector:

    def __init__(self):
        self.center_words = []
        self.context_words = []
        self.doc_ids = []

    def prepare_dataset(data):
        # assume data is the pandas dataframe with pid to claims mapping

        # parse dataframe to get document mapping
        documents = dict()
        words = []
        for name, claims in data.groupby("PATENT_ID"):
            documents[name] = []
            for claim in claims:
                documents[name].extend(claim.split())
            words.extend(documents[name])
        # data sharing
        self.documents = documents
        # use words to generate dictionary
        w_count = [["unknow", -1]]
        w_count.extend(collections.Counter(words).most_common(vocab_size-1))
        # num input sample equal num of words
        self.instance_num = len(words)
        del words
        dictionary = dict()
        for word, _ in w_count:
            dictionary[word] = len(dictionary)


        for pid, doc in documents.items():
            documents[pid] = [vocab_size]*window_size + [dictionary[w] for w in doc] + [vocab_size]*window_size
            # start for center words
            for i in range(window_size, len(doc)-window_size):
                yield (documents[pid][i], pid, documents[pid][i-window_size:i]+documents[pid][i+1:i+window_size+1])
                #self.center_words.append(documents[pid][i])
                #self.doc_ids.append(pid)
                #self.context_words.append(documents[pid][i-window_size:i]+documents[pid][i+1:i+window_size+1])

        # clean up
        # shuffle data

    def generate_bacth(self):
        cur_iter = 0
        center_words = []
        doc_ids = []
        context_words = []
        for data in self.prepare_dataset():
            if(cur_iter == BATCH_SIZE):
                yield(center_words, doc_ids, context_words)
                center_words = [data[0]]
                doc_ids = [data[1]]
                context_words = [data[2]]
            else:
                center_words.append(data[0])
                doc_ids.append(data[1])
                context_words = data[2]
        # left over
        yield (center_words, doc_ids, context_words)



    def build_graph(self):

        graph = tf.Graph()
        with graph.as_default():

            with tf.name_scope("input_layer"):
                doc_ids_input = tf.placeholder(tf.int32, shape=[batch_size])
                context_word_input = tf.placeholder(tf.int32, shape=[batch_size*context_window_size])
                center_word_input = tf.placeholder(tf.int32, shape=[batch_size, 1])

            with tf.name_scope("hidden_layer"):
                # create emebedding weights
                word_embed_w = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
                word_embed_w = tf.concat([word_embed_w, tf.zeros((1, embedding_size))], 0)
                doc_embed_w = tf.Variable(tf.random_uniform([len(self.documents), embedding_size], -1.0, 1.0))

            with tf.name_scope("loss_layer"):

                # create embeeding lookup
                segment_ids = tf.constant(np.repeat(np.arange(batch_size), context_window_size), dtype=tf.int32)
                word_embed_lookup = tf.segment_mean(tf.nn.emebedding_lookup(word_embed_w, center_word_input), segment_ids)
                doc_emebed_lookup = tf.nn.embedding_lookup(doc_embed_w, doc_ids_input)
                embedding = (word_embed_lookup + doc_emebed_lookup)/2.0

                # nce_loss
                nce_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0/np.sqrt(embedding_size)))
                nce_bias = tf.Varaible(tf.zeros([vocab_size]))

                nce_loss = tf.reduce_mean(tf.nn.nce_loss(nce_w, nce_bias, center_word_input, embedding, num_negative_sample, vocab_size))




            # start training

            epoch = 5
            optimizer = tf.trian.AdagradOptimizer(0.5).minimize(nce_loss)

            # normalize doc embedding
            norm = tf.sqrt(tf.reduce_sum(tf.square(doc_embed_w), 1, keep_dims=True))
            normalized_doc_embedding = doc_embed_w/norm

            with tf.Seesion(graph=graph) as session:
                tf.global_variables_initializer().run()
                print("start training")
                for i in range(epoch):
                    print("epoch: ", i)
                    # batch Training
                    for batch in self.generate_bacth():
                        _, batch_loss = session.run([optimizer, nce_loss], feed_dict=
                        {doc_ids_input: np.squeeze(batch[0]),
                        context_word_input: np.squeeze(batch[1]),
                        center_word_input: np.squeeze(batch[2])
                        })
                        print("loss: ", batch_loss)

                # save model
                saver = tf.train.Saver()
                saver.save(session, "stored_session.ckpt")

            # save weights/word embedding
            final_word_embedding = word_embed_w.eval()
            final_doc_embedding = normalized_doc_embedding.eval()
