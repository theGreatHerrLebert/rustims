import tensorflow as tf


class GRUChargeStatePredictor(tf.keras.models.Model):

    def __init__(self,
                 num_tokens,
                 max_charge=4,
                 seq_len=50,
                 emb_dim=128,
                 gru_1=128,
                 gru_2=64,
                 rdo=0.0,
                 do=0.2):
        super(GRUChargeStatePredictor, self).__init__()

        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=emb_dim, input_length=seq_len)

        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True,
                                                                      name='GRU1'))

        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False,
                                                                      name='GRU2',
                                                                      recurrent_dropout=rdo))

        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name='Dense1',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        self.dense2 = tf.keras.layers.Dense(64, activation='relu', name='Dense2',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        self.dropout = tf.keras.layers.Dropout(do, name='Dropout')

        self.out = tf.keras.layers.Dense(max_charge, activation='softmax', name='Output')

    def call(self, inputs, **kwargs):
        """
        :param inputs: should contain: (sequence)
        """
        # get inputs
        seq = inputs
        # sequence learning
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))
        # regularize
        d1 = self.dropout(self.dense1(x_recurrent))
        # output
        return self.out(self.dense2(d1))
