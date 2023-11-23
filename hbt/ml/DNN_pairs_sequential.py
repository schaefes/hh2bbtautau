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


def dict_get(d):
    return lambda x: d[x]


class CreatePairsSum(tf.keras.layers.Layer):
    def call(self, inputs, dict_vals, jet_num):
        dtype_agg = inputs.dtype
        jet_num = tf.cast(jet_num, dtype=tf.int32)
        pair_idx = tf.gather(dict_vals, jet_num - 2)
        mask = tf.where(pair_idx == -1, False, True)
        pair_idx = tf.where(pair_idx == -1, tf.constant([0], dtype=pair_idx.dtype)[0], pair_idx)
        pairs = tf.gather(inputs, pair_idx, batch_dims=1)
        mask = mask[:, :, 0]
        pairs_ragged = tf.ragged.boolean_mask(pairs, mask)
        pairs_sum = tf.reduce_sum(pairs_ragged, axis=2)

        # get the number of pairs and shape/cast the tensor accordingly for the aggregation func
        pairs_num = tf.math.reduce_sum(tf.where(mask, 1, 0), axis=1)
        pairs_num = tf.expand_dims(pairs_num, axis=1)
        pairs_num = tf.cast(pairs_num, dtype=dtype_agg)

        return pairs_sum, pairs_num


class CreatePairsConcat(tf.keras.layers.Layer):
    def call(self, inputs, dict_vals, jet_num):
        dtype_agg = inputs.dtype
        jet_num = tf.cast(jet_num, dtype=tf.int32)
        pair_idx = tf.gather(dict_vals, jet_num - 2)
        mask = tf.where(pair_idx == -1, False, True)
        pair_idx = tf.where(pair_idx == -1, tf.constant([0], dtype=pair_idx.dtype)[0], pair_idx)
        pairs = tf.gather(inputs, pair_idx, batch_dims=1)
        pairs_reversed = tf.reverse(pairs, [-2])
        mask = mask[:, :, 0]
        pairs_ragged = tf.ragged.boolean_mask(pairs, mask)
        pairs_reversed_ragged = tf.ragged.boolean_mask(pairs_reversed, mask)
        pairs_concat = tf.concat((pairs_ragged, pairs_reversed_ragged), axis=1)
        # concat along pair axis
        pairs_concat = tf.concat((pairs_concat[:, :, 0, :], pairs_concat[:, :, 1, :]), axis=2)
        # get the number of pairs and shape/cast the tensor accordingly for the aggregation func
        # factor 2 due to concatenation
        pairs_num = tf.math.reduce_sum(tf.where(mask, 1, 0), axis=1)
        pairs_num = tf.expand_dims(pairs_num, axis=1)
        pairs_num = 2 * tf.cast(pairs_num, dtype=dtype_agg)

        return pairs_concat, pairs_num


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

    def __init__(self, nodes, activation, deepset=False):
        super(DenseBlock, self).__init__()
        self.deepset = deepset

        if self.deepset:
            layers = self.timedistributed_dense_block(nodes, activation)
        else:
            layers = self.dense_block(nodes, activation)
        self.dense, self.batchnorm, self.activation = layers

    def dense_block(self, nodes, activation):
        name_prefix = 'FeedForward/DenseBlock/'

        layers = (tf.keras.layers.Dense(nodes, use_bias=False,
                                        name=name_prefix + 'Dense'),
                  tf.keras.layers.BatchNormalization(
                      name=name_prefix + 'BatchNorm'),
                  tf.keras.layers.Activation(
                      activation, name=name_prefix + activation),
                  )
        return layers

    def timedistributed_dense_block(self, nodes, activation):
        name_prefix = 'DeepSetPS/DenseBlock/'

        layers = (tf.keras.layers.Dense(nodes, use_bias=False,
                                        name=name_prefix + 'Dense'),
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

    def __init__(self, nodes, activations, aggregations, masking_val, mean, std, dict_vals):
        super(DeepSet, self).__init__()
        self.aggregations = aggregations
        self.masking_val = masking_val
        self.hidden_layers1 = [DenseBlock(node, activation, deepset=True)
                              for node, activation in zip(nodes, activations)]
        self.hidden_layers2 = [DenseBlock(node, activation, deepset=True)
                              for node, activation in zip(nodes, activations)]

        # mean and std for standardization
        self.mean = mean
        self.std = std

        # aggregation layers
        self.sum_layer = SumLayer()
        self.max_layer = MaxLayer()
        self.min_layer = MinLayer()
        self.mean_layer = MeanLayer()
        self.var_layer = VarLayer()
        self.std_layer = StdLayer()
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)

        # masking layer
        self.masking_layer = tf.keras.layers.Masking(mask_value=masking_val)

        # pair craetion layer
        self.create_pairs_concat = CreatePairsConcat()

        # index dictionary for the pair indices
        self.dict_vals = dict_vals

    def call(self, inputs):
        # compute mask based on he masking value in inputs
        mask = self.masking_layer.compute_mask(inputs)
        inputs = (inputs - self.mean) / self.std
        inputs_masked = tf.ragged.boolean_mask(inputs, mask)

        x = self.hidden_layers1[0](inputs_masked)
        for layer in self.hidden_layers1[1:]:
            x = layer(x)

        # get number of jets per event, required for the mean aggregation layer
        mask_jet_num = tf.where(mask, 1., 0.)
        jet_num = tf.convert_to_tensor(tf.math.reduce_sum(mask_jet_num, axis=1), dtype=x.dtype)

        pairs, pairs_num = self.create_pairs_concat(x, self.dict_vals, jet_num)
        jet_num = tf.expand_dims(jet_num, axis=1)

        y = self.hidden_layers2[0](pairs)
        for layer in self.hidden_layers2[1:]:
            y = layer(y)

        # aggregations
        aggregations = []
        for inp, obj_num in ((x, jet_num), (y, pairs_num)):
            aggregation_layers = []
            for func in self.aggregations:
                if func == "Sum":
                    sum_layer = self.sum_layer(inp)
                    aggregation_layers.append(sum_layer)
                elif func == "Max":
                    max_layer = self.max_layer(inp)
                    aggregation_layers.append(max_layer)
                elif func == "Min":
                    min_layer = self.min_layer(inp)
                    aggregation_layers.append(min_layer)
                elif func == "Mean":
                    mean_layer = self.mean_layer(inp, obj_num)
                    aggregation_layers.append(mean_layer)
                elif func == "Var":
                    var_layer = self.var_layer(inp)
                    aggregation_layers.append(var_layer)
                elif func == "Std":
                    std_layer = self.std_layer(inp)
                    aggregation_layers.append(std_layer)
            concat_aggregation = self.concat_layer(aggregation_layers)
            aggregations.append(concat_aggregation)

        # concat all aggregations
        output = aggregations

        return output


class FeedForwardNetwork(tf.keras.Model):

    def __init__(self, nodes, activations, n_classes, mean, std, combined=False):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_layers = [DenseBlock(node, activation, deepset=False)
                              for node, activation in zip(nodes, activations)]
        self.output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')

        # mean and std for standardization
        self.mean = mean
        self.std = std
        self.len_mean = len(mean)

        # Decides on the mean and std in call, if true padding of 0 and 1 in mean and std on DS op
        self.combined = combined

    def call(self, inputs):
        # Padding of 0 and 1 ind mean and std, these are applied on the DS input nodes
        if self.combined:
            mean = tf.zeros(inputs.shape[1] - self.len_mean)
            std = tf.ones_like(mean)
            mean = tf.concat((mean, self.mean), axis=0)
            std = tf.concat((std, self.std), axis=0)
            inputs = (inputs - mean) / std
        else:
            inputs = (inputs - self.mean) / self.std

        x = self.hidden_layers[0](inputs)
        # print('ff input:', inputs.shape)
        for layer in self.hidden_layers[1:]:
            x = layer(x)
        output = self.output_layer(x)
        # print('ff output:', output.shape)
        return output


class CombinedDeepSetPairsSequentialNetwork(tf.keras.Model):
    def __init__(self, deepset_config, feedforward_config):
        super(CombinedDeepSetPairsSequentialNetwork, self).__init__()

        self.deepset_network = DeepSet(**deepset_config)
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.combied = True
        self.feed_forward_network = FeedForwardNetwork(**feedforward_config)

    def call(self, inputs):
        deepset_inputs, feedforward_inputs = inputs
        deepset_output = self.deepset_network(deepset_inputs)
        deepset_output = self.concat_layer(deepset_output)
        concatenated_inputs = self.concat_layer((deepset_output, feedforward_inputs))
        output = self.feed_forward_network(concatenated_inputs)
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

    comb = CombinedDeepSetPairsSequentialNetwork(deepset_config, feedforward_config)
    from IPython import embed
    embed()
