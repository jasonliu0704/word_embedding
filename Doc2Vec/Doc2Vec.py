import pandas as pd
import numpy as np
import collections
import tensorflow as tf

batch_size = 128
embedding_size = 50 # Dimension of the embedding vector.
window_size = 5
context_window_size = 2*window_size
num_negative_sample = 5


class ParagraphVector:

    def __init__(self, minCount = 0):
        self.center_words = []
        self.context_words = []
        self.doc_ids = []
        self.doc_generated = False
        self.minCount = minCount

    def prepare_dataset(self, data):
        # assume data is the pandas dataframe with pid to claims mapping

        # parse dataframe to get document mapping
        # tf.nn.emebedding_lookup need doc_id starting from 0
        documents = dict()
        words = []

        # create a claim doc id to doc id mapping
        doc2id_map = dict()
        id2doc_map = dict()
        for name in data["PATENT_ID"]:
            if name not in doc2id_map:
                doc2id_map[name] = len(doc2id_map)
                id2doc_map[doc2id_map[name]] = name
        self.id2doc_map = id2doc_map
        self.doc2id_map = doc2id_map

        # need to clean the claim data
        # claim Number

        # generate docuement and word from data source
        for name, claims in data.groupby("PATENT_ID"):
            doc_name = doc2id_map[name]
            documents[doc_name] = []
            for claim in claims["CLAIM_TEXT"]:
                documents[doc_name].extend(claim.split())
            words.extend(documents[doc_name])
        # save document mapping
        import pickle
        pickle.dump(self.doc2id_map, open("doc2id_map.p", "wb"))
        #del self.doc2id_map
        del self.id2doc_map

        # data sharing
        self.documents = documents

        # use words to generate dictionary
        # w_count = [["unknow", -1]]

        # num input sample equal num of words
        self.instance_num = len(words)
        # handle dictionary size with min count
        w_count = collections.Counter(words)
        del words
        # generate word dicitonary
        dictionary = dict()
        dictionary["unknow"] = 0
        for word, c in w_count.items():
            if(c > self.minCount):
                dictionary[word] = len(dictionary)
        self.dictionary = dictionary
        self.vocab_size = len(dictionary)

    def generate_document(self):
        for pid, doc in self.documents.items():
            if(self.doc_generated is False):
                # handle unknow words
                self.documents[pid] = [self.vocab_size]*window_size +[self.dictionary[w] if w in self.dictionary else 0 for w in doc] + [self.vocab_size]*window_size
            # start for center words
            for i in range(window_size, len(doc)-window_size):
                yield (self.documents[pid][i], pid, self.documents[pid][i-window_size:i]+self.documents[pid][i+1:i+window_size+1])
                #self.center_words.append(documents[pid][i])
                #self.doc_ids.append(pid)
                #self.context_words.append(documents[pid][i-window_size:i]+documents[pid][i+1:i+window_size+1])

        # clean up
        # shuffle data

    def generate_batch(self):
        cur_iter = 0
        center_words = []
        doc_ids = []
        context_words = []
        for data in self.generate_document():
            if(cur_iter == batch_size):
                yield(center_words, doc_ids, context_words)
                center_words = [data[0]]
                doc_ids = [data[1]]
                context_words = data[2]
                cur_iter = 0
            else:
                center_words.append(data[0])
                doc_ids.append(data[1])
                context_words.extend(data[2])

            cur_iter += 1
        # ditch the left over
        #yield (center_words, doc_ids, context_words)
        self.doc_generated = True

    def infer(self, doc_name):
        return self.final_doc_embedding[self.doc2id_map[doc_name]]

    def build_graph(self):

        graph = tf.Graph()
        with graph.as_default():

            with tf.name_scope("input_layer"):
                doc_ids_input = tf.placeholder(tf.int32, shape=[batch_size])
                context_word_input = tf.placeholder(tf.int32, shape=[batch_size*context_window_size])
                center_word_input = tf.placeholder(tf.int32, shape=[batch_size, 1])

            with tf.name_scope("hidden_layer"):
                # create emebedding weights
                word_embed_w = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
                word_embed_w = tf.concat([word_embed_w, tf.zeros((1, embedding_size))], 0)
                doc_embed_w = tf.Variable(tf.random_uniform([len(self.documents), embedding_size], -1.0, 1.0))

            with tf.name_scope("loss_layer"):

                # create embeeding lookup
                segment_ids = tf.constant(np.repeat(np.arange(batch_size), context_window_size), dtype=tf.int32)
                word_embed_lookup = tf.segment_mean(tf.nn.embedding_lookup(word_embed_w, context_word_input), segment_ids)
                doc_emebed_lookup = tf.nn.embedding_lookup(doc_embed_w, doc_ids_input)
                embedding = (word_embed_lookup + doc_emebed_lookup)/2.0

                # nce_loss
                nce_w = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size], stddev=1.0/np.sqrt(embedding_size)))
                nce_bias = tf.Variable(tf.zeros([self.vocab_size]))

                nce_loss = tf.reduce_mean(tf.nn.nce_loss(nce_w, nce_bias, center_word_input, embedding, num_negative_sample, self.vocab_size))




            # start training

            epoch = 2
            optimizer = tf.train.AdagradOptimizer(0.5).minimize(nce_loss)

            # normalize doc embedding
            norm = tf.sqrt(tf.reduce_sum(tf.square(doc_embed_w), 1, keep_dims=True))
            normalized_doc_embedding = doc_embed_w/norm

            with tf.Session(graph=graph) as session:
                tf.global_variables_initializer().run()
                print("start training")
                for i in range(epoch):
                    print("epoch: ", i)
                    # batch Training
                    for batch in self.generate_batch():
                        #print (np.reshape(batch[0], (batch_size,1)),batch[1],batch[2])
                        _, batch_loss = session.run([optimizer, nce_loss], feed_dict=
                        {center_word_input: np.reshape(batch[0], (batch_size,1)),
                        doc_ids_input:batch[1],
                        context_word_input: batch[2]
                        })
                        print("loss: ", batch_loss)

                # save model
                saver = tf.train.Saver()
                saver.save(session, "./stored_session.ckpt")

                # save weights/word embedding
                self.final_word_embedding = word_embed_w.eval(session=session)
                self.final_doc_embedding = normalized_doc_embedding.eval(session=session)
