import tensorflow as tf
import numpy as np


# Define Custom Aggregation Functions for deep Sets

class SumLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        sum_inputs = tf.math.reduce_sum(inputs, axis=1, keepdims=False)
        return sum_inputs


class MaxLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        max_inputs = tf.math.reduce_max(inputs, axis=1, keepdims=False)
        return max_inputs


class MinLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        min_inputs = tf.math.reduce_min(inputs, axis=1, keepdims=False)
        return min_inputs


class MeanLayer(tf.keras.layers.Layer):
    def call(self, inputs, jet_num):
        mean_inputs = tf.math.reduce_sum(inputs, axis=1, keepdims=False) / jet_num
        return mean_inputs


class VarLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        variance_inputs = tf.math.reduce_variance(inputs, axis=1, keepdims=False)
        return variance_inputs


class StdLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        std_inputs = tf.math.reduce_std(inputs, axis=1, keepdims=False)
        return std_inputs


class InpMasking(tf.keras.layers.Layer):
    def call(self, inputs, masking_val):
        print('START')
        mask = (inputs != masking_val)
        print('Mask Shape', mask.shape)
        print('Input Shape', inputs.shape)
        masked_inp = inputs[mask]
        print('Masked Inp', masked_inp.shape)
        return masked_inp


class ShapMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, slice_idx):
        super(ShapMaskingLayer, self).__init__()
        self.slice_idx = slice_idx

    def call(self, inputs):
        # the slice idx is given by the numer of Input Features for the Deep Sets
        deepSets_inp = inputs[:, :, :self.slice_idx]
        events_inp = inputs[:, 0, self.slice_idx:]
        return [deepSets_inp, events_inp]


class Dummy_2(tf.keras.Model):
    def __init__(self):
        super(Dummy_2, self).__init__()
        self.Input = tf.keras.layers.Input(shape=(None, 10))
        self.dense = tf.keras.layers.Dense(20)

    def call(self, inputs):
        x = self.dense(inputs)

        return x


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, nodes, activation, n_l2, deepset=False, inp_type=None):
        super(DenseBlock, self).__init__()
        self.deepset = deepset
        self.inp_type = inp_type

        if self.deepset:
            layers = self.timedistributed_dense_block(nodes, activation, n_l2, inp_type)
        else:
            layers = self.dense_block(nodes, activation, n_l2)
        self.dense, self.batchnorm, self.activation = layers

    def dense_block(self, nodes, activation, n_l2):
        name_prefix = 'FeedForward/DenseBlock/'
        l2 = tf.keras.regularizers.l2(n_l2)
        layers = (tf.keras.layers.Dense(nodes, use_bias=False,
                                        name=name_prefix + 'Dense',
                                        kernel_regularizer=l2),
                  tf.keras.layers.BatchNormalization(
                      name=name_prefix + 'BatchNorm'),
                  tf.keras.layers.Activation(
                      activation, name=name_prefix + activation),
                  )
        return layers

    def timedistributed_dense_block(self, nodes, activation, n_l2, inp_type):
        name_prefix = f'DeepSet{inp_type}/DenseBlock/'
        l2 = tf.keras.regularizers.l2(n_l2)
        layers = (tf.keras.layers.Dense(nodes, use_bias=False,
                                        name=name_prefix + 'Dense',
                                        kernel_regularizer=l2),
                  tf.keras.layers.BatchNormalization(
                      name=name_prefix + 'BatchNorm'),
                  tf.keras.layers.Activation(
                      activation, name=name_prefix + activation)
                  )
        layers = [tf.keras.layers.TimeDistributed(layer) for layer in layers]
        return layers

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batchnorm(x)
        output = self.activation(x)
        return output


class DeepSet(tf.keras.Model):

    def __init__(self, nodes, activations, n_l2, aggregations, masking_val, inp_type, mean, std,
                 ff_mean, ff_std, event_to_jet):
        super(DeepSet, self).__init__()
        self.aggregations = aggregations
        self.masking_val = masking_val
        self.inp_type = inp_type
        self.hidden_layers = [DenseBlock(node, activation, n_l2, deepset=True, inp_type=self.inp_type)
                              for node, activation in zip(nodes, activations)]

        # aggregation layers
        self.sum_layer = SumLayer()
        self.max_layer = MaxLayer()
        self.min_layer = MinLayer()
        self.mean_layer = MeanLayer()
        self.var_layer = VarLayer()
        self.std_layer = StdLayer()
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)

        # mean and std values
        self.mean = mean
        self.std = std
        self.ff_mean = ff_mean
        self.ff_std = ff_std

        # decide on information given to ds, if true: event level info for each jet
        self.event_to_jet = event_to_jet

        # masking layer
        self.masking_layer = tf.keras.layers.Masking(mask_value=masking_val)

    def call(self, inputs):
        # compute mask based on he masking value in inputs
        mask = self.masking_layer.compute_mask(inputs)
        inputs = (inputs - self.mean) / self.std
        inputs_masked = tf.ragged.boolean_mask(inputs, mask)

        x = self.hidden_layers[0](inputs_masked)
        for layer in self.hidden_layers[1:]:
            x = layer(x)
        # get number of jets per event, required for the mean aggregation layer
        mask_jet_num = tf.where(mask, 1., 0.)
        jet_num = tf.convert_to_tensor(tf.math.reduce_sum(mask_jet_num, axis=1), dtype=x.dtype)
        jet_num = tf.expand_dims(jet_num, axis=1)

        # aggregations
        aggregation_layers = []
        for func in self.aggregations:
            if func == "Sum":
                sum_layer = self.sum_layer(x)
                aggregation_layers.append(sum_layer)
            elif func == "Max":
                max_layer = self.max_layer(x)
                aggregation_layers.append(max_layer)
            elif func == "Min":
                min_layer = self.min_layer(x)
                aggregation_layers.append(min_layer)
            elif func == "Mean":
                mean_layer = self.mean_layer(x, jet_num)
                aggregation_layers.append(mean_layer)
            elif func == "Var":
                var_layer = self.var_layer(x)
                aggregation_layers.append(var_layer)
            elif func == "Std":
                std_layer = self.std_layer(x)
                aggregation_layers.append(std_layer)

        # concat all aggregations
        output = self.concat_layer(aggregation_layers)

        return output


class FeedForwardNetwork(tf.keras.Model):

    def __init__(self, nodes, activations, n_l2, n_classes, mean, std, combined=False):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_layers = [DenseBlock(node, activation, n_l2, deepset=False)
                              for node, activation in zip(nodes, activations)]
        self.output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')
        self.mean = mean
        self.std = std
        self.len_mean = len(mean)

        self.combined = combined

    def call(self, inputs):
        if self.combined:
            mean = tf.zeros(inputs.shape[1] - self.len_mean)
            std = tf.ones_like(mean)
            mean = tf.concat((mean, self.mean), axis=0)
            std = tf.concat((std, self.std), axis=0)
            inputs = (inputs - mean) / std
        else:
            inputs = (inputs - self.mean) / self.std

        x = self.hidden_layers[0](inputs)

        for layer in self.hidden_layers[1:]:
            x = layer(x)
        output = self.output_layer(x)

        return output


class CombinedDeepSetPairsParallelNetwork(tf.keras.Model):
    def __init__(self, deepset_jets_config, deepset_pairs_config, feedforward_config):
        super(CombinedDeepSetPairsParallelNetwork, self).__init__()

        self.deepset_jets_network = DeepSet(**deepset_jets_config)
        self.deepset_pairs_network = DeepSet(**deepset_pairs_config)
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.feed_forward_network = FeedForwardNetwork(**feedforward_config)

    def call(self, inputs):
        deepset_jets_inputs, deepset_pairs_inputs, feedforward_inputs = inputs
        deepset_jets_output = self.deepset_jets_network(deepset_jets_inputs)
        deepset_pairs_output = self.deepset_pairs_network(deepset_pairs_inputs)
        concatenated_inputs = self.concat_layer((deepset_jets_output, deepset_pairs_output, feedforward_inputs))
        output = self.feed_forward_network(concatenated_inputs)
        return output


class BaseLineFFPairs(tf.keras.Model):
    def __init__(self, feedforward_config):
        super(BaseLineFFPairs, self).__init__()

        self.feed_forward_network = FeedForwardNetwork(**feedforward_config)

    def call(self, inputs):
        output = self.feed_forward_network(inputs)
        return output


class ShapMasking(tf.keras.Model):
    def __init__(self, slice_idx, model):
        super(ShapMasking, self).__init__()
        self.slice_idx = slice_idx
        self.model = model
        self.masking_layer = ShapMaskingLayer(slice_idx)

    def call(self, inputs):
        masking_output = self.masking_layer(inputs)
        output = self.model(masking_output)

        return output


if __name__ == '__main__':
    deep_set_input = tf.ones(shape=(2, 3, 2))
    feed_forward_input = tf.ones(shape=(2, 32))

    deepset_config = {'nodes': (20, 10, 1), 'activations': ('relu', 'relu', 'selu')}
    feedforward_config = {'nodes': (128, 128, 128), 'activations': (
        'selu', 'selu', 'selu'), 'n_classes': 3}

    comb = CombinedDeepSetPairsParallelNetwork(deepset_config, feedforward_config)
