import tensorflow as tf
from imspy.algorithm.ccs.predictors import SquareRootProjectionLayer

@tf.keras.saving.register_keras_serializable()
class GRUCCSPredictorStd(tf.keras.models.Model):
    """
    Deep Learning model combining initial linear fit with sequence-based features, both scalar and complex.
    Now includes a shared backbone and calibrated variance prediction.
    """

    def __init__(self, slopes, intercepts, num_tokens,
                 max_peptide_length=50,
                 emb_dim=128,
                 gru_1=128,
                 gru_2=64,
                 rdo=0.0,
                 do=0.2):
        super(GRUCCSPredictorStd, self).__init__()
        self.max_peptide_length = max_peptide_length

        # Scalar feature projection layer
        self.initial = SquareRootProjectionLayer(slopes, intercepts)

        # Sequence feature layers
        self.emb = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=emb_dim)
        self.gru1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_1, return_sequences=True, name='GRU1'))
        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_2, return_sequences=False,
                                                                      name='GRU2',
                                                                      recurrent_dropout=rdo))

        # Shared dense layers
        self.shared_dense1 = tf.keras.layers.Dense(128, activation='relu',
                                                   kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.shared_dense2 = tf.keras.layers.Dense(64, activation='relu',
                                                   kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        # Separate branches
        self.mean_dense = tf.keras.layers.Dense(64, activation='relu',
                                                kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))
        self.std_dense = tf.keras.layers.Dense(64, activation='relu',
                                               kernel_regularizer=tf.keras.regularizers.l1_l2(1e-3, 1e-3))

        # Outputs
        self.out_mean = tf.keras.layers.Dense(1, activation=None)
        self.out_var = tf.keras.layers.Dense(1, activation=tf.nn.softplus)  # Ensure positivity for variance

        # Dropout
        self.dropout = tf.keras.layers.Dropout(do)

    def call(self, inputs, training=False):
        """
        :param inputs: should contain: (mz, charge_one_hot, seq_as_token_indices)
        """
        mz, charge, seq = inputs[0], inputs[1], inputs[2]

        # Sequence learning
        x_recurrent = self.gru2(self.gru1(self.emb(seq)))

        # Concatenate scalar and sequence features
        concat = tf.keras.layers.Concatenate()([charge, x_recurrent])

        # Shared layers
        shared = self.shared_dense2(self.shared_dense1(concat))

        # Mean branch
        mean_output = self.out_mean(self.mean_dense(shared))

        # Variance branch
        var_output = self.out_var(self.std_dense(shared))

        # Scalar projection
        initial_output = self.initial([mz, charge])

        # Final outputs
        final_mean = initial_output + mean_output
        return final_mean, var_output, mean_output

    def get_config(self):
        config = super(GRUCCSPredictorStd, self).get_config()
        config.update({
            'slopes': self.initial.slopes_init,
            'intercepts': self.initial.intercepts_init,
            'num_tokens': self.emb.input_dim - 1,
            'max_peptide_length': self.max_peptide_length,
            'emb_dim': self.emb.output_dim,
            'gru_1': self.gru1.forward_layer.units,
            'gru_2': self.gru2.forward_layer.units,
            'rdo': self.gru2.forward_layer.recurrent_dropout,
            'do': self.dropout.rate,

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
