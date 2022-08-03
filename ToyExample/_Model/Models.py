#models
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
# from _Model.Proxy import *

regularizer = tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00001)
# regularizer = tf.keras.regularizers.l1_l2(l1=0.0000005, l2=0.000005) # the best

# mindt = 1e-4
# maxdt = 1.0
class WiseMultiply(tf.keras.layers.Layer):
    def __init__(self, kernel=None, **kwargs):
        self.kernel = kernel
        super(WiseMultiply, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.kernel is None:
            shape = [1]
            for i in range(1, len(input_shape)):
                shape.append(input_shape[i])
            self.shape = tuple(shape)
            kernel = self.add_weight(shape=self.shape, initializer='uniform',
                                     constraint=tf.keras.constraints.NonNeg(),
                                     regularizer=tf.keras.regularizers.l1_l2(l1=0.00005, l2=0.00005))
            self.kernel = kernel

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape


class Phy_GRU_cell(layers.AbstractRNNCell):

    def __init__(self,
                 units,
                 activation=tf.sigmoid,
                 recurrent_activation=tf.tanh,
                 kernel_regularizer=regularizer,
                 recurrent_regularizer=regularizer,
                 bias_regularizer=regularizer,
                 dt_regularizer=regularizer,
                 mindt = 1e-4, 
                 maxdt = 1.0,
                 **kwargs):
        self.units = units[0]
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.dt_regularizer = dt_regularizer
        self.mindt = mindt
        self.maxdt = maxdt
        super(Phy_GRU_cell, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "dt_regularizer": self.dt_regularizer,
            "mindt": self.mindt,
            "maxdt": self.maxdt,
        })
        return config

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):

        nfeature = input_shape[-1]

        # kernel:
        self.input_kernel = self.add_weight(shape=(nfeature, 3 * self.units), initializer='uniform',
                                            regularizer=self.kernel_regularizer,
                                            name='input_kernel')
        # recurrent kernel:
        self.recurrent_kernel = self.add_weight(shape=(self.units, 3 * self.units), initializer='uniform',
                                                regularizer=self.recurrent_regularizer,
                                                name='recurrent_kernel')
        # bias:
        self.biases = self.add_weight(shape=(2, 3 * self.units),
                                      initializer=tf.keras.initializers.Zeros(),
                                      regularizer=self.bias_regularizer,
                                      name='biases')
        # delta time:
        self.dt = self.add_weight(shape=(1,), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=self.mindt, max_value=self.maxdt, rate=1.0, axis=0),
                                  regularizer=self.dt_regularizer,
                                  name='dt')

        self.built = True

    def call(self, cell_inputs, cell_states):

        # extract inputs:
        h_tm1 = cell_states[0]
        input_bias, recurrent_bias = tf.unstack(self.biases)

        # inputs projected by all gate matrices at once
        matrix_x = backend.dot(cell_inputs, self.input_kernel)
        matrix_x = backend.bias_add(matrix_x, input_bias)

        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)

        # hidden state projected by all gate matrices at once
        matrix_h = backend.dot(h_tm1, self.recurrent_kernel)
        matrix_h = backend.bias_add(matrix_h, recurrent_bias)

        recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_h, 3, axis=1)

        z = self.activation(x_z + recurrent_z)
        r = self.activation(x_r + recurrent_r)
        hh = self.recurrent_activation(x_h + r * recurrent_h)

        # previous and candidate state mixed by update gate
        h = h_tm1 + z * hh * self.dt

        return h, [h]


class Phy_RNN_cell(layers.AbstractRNNCell):

    def __init__(self,
                 units,
                 recurrent_activation=tf.tanh,
                 kernel_regularizer=regularizer,
                 recurrent_regularizer=regularizer,
                 bias_regularizer=regularizer,
                 dt_regularizer=regularizer,
                 mindt = 1e-4, 
                 maxdt = 1.0,
                **kwargs):

        self.units = units[0]
        self.recurrent_activation = recurrent_activation
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.dt_regularizer = dt_regularizer
        self.mindt = mindt
        self.maxdt = maxdt

        super(Phy_RNN_cell, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "recurrent_activation": self.recurrent_activation,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "dt_regularizer": self.dt_regularizer,
            "mindt": self.mindt,
            "maxdt": self.maxdt 
        })
        return config

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):

        nfeature = input_shape[-1]

        # kernel:
        self.input_kernel = self.add_weight(shape=(nfeature, 1 * self.units), initializer='uniform',
                                            regularizer=self.kernel_regularizer,
                                            name='input_kernel')
        # recurrent kernel:
        self.recurrent_kernel = self.add_weight(shape=(self.units, 1 * self.units), initializer='uniform',
                                                regularizer=self.recurrent_regularizer,
                                                name='recurrent_kernel')
        # bias:
        self.biases = self.add_weight(shape=(2, 1 * self.units), initializer=tf.keras.initializers.Zeros(),
                                      regularizer=self.bias_regularizer,
                                      name='biases')
        # delta time:
        self.dt = self.add_weight(shape=(1, 1), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=self.mindt, max_value=self.maxdt, rate=1.0, axis=0),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000001, l2=0.000001),
                                  name='dt')
        self.built = True

    def call(self, cell_inputs, cell_states):
        # extract inputs:
        h_tm1 = cell_states[0]
        input_bias, recurrent_bias = tf.unstack(self.biases)

        # inputs projected by all gate matrices at once
        matrix_x = backend.dot(cell_inputs, self.input_kernel)
        matrix_x = backend.bias_add(matrix_x, input_bias)

        # hidden state projected by all gate matrices at once
        matrix_h = backend.dot(h_tm1, self.recurrent_kernel)
        matrix_h = backend.bias_add(matrix_h, recurrent_bias)
        hh = self.recurrent_activation(matrix_x + matrix_h)

        # previous and candidate state mixed by update gate
        h = h_tm1 + hh * self.dt

        return h, [h]


class Phy_GRUa_cell(layers.AbstractRNNCell):

    def __init__(self,
                 units,
                 activation=tf.sigmoid,
                 recurrent_activation=tf.tanh,
                 kernel_regularizer=regularizer,
                 recurrent_regularizer=regularizer,
                 bias_regularizer=regularizer,
                 dt_regularizer=regularizer,
                 mindt = 1e-4, 
                 maxdt = 1.0,
                 **kwargs):
        self.units = units[0]
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.dt_regularizer = dt_regularizer
        self.mindt = mindt
        self.maxdt = maxdt
        super(Phy_GRUa_cell, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "dt_regularizer": self.dt_regularizer,
            "mindt": self.mindt,
            "maxdt": self.maxdt,
        })
        return config

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):

        nfeature = input_shape[-1]

        # kernel:
        self.input_kernel = self.add_weight(shape=(nfeature, 3 * self.units), initializer='uniform',
                                            regularizer=self.kernel_regularizer,
                                            name='input_kernel')
        # recurrent kernel:
        self.recurrent_kernel = self.add_weight(shape=(self.units, 3 * self.units), initializer='uniform',
                                                regularizer=self.recurrent_regularizer,
                                                name='recurrent_kernel')
        # bias:
        self.biases = self.add_weight(shape=(2, 3 * self.units),
                                      initializer=tf.keras.initializers.Zeros(),
                                      regularizer=self.bias_regularizer,
                                      name='biases')
        # delta time:
        self.dt = self.add_weight(shape=(1,), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=self.mindt, max_value=self.maxdt, rate=1.0, axis=0),
                                  regularizer=self.dt_regularizer,
                                  name='dt')

        self.built = True

    def call(self, cell_inputs, cell_states):

        # extract inputs:
        h_tm1 = cell_states[0]
        input_bias, recurrent_bias = tf.unstack(self.biases)

        # inputs projected by all gate matrices at once
        matrix_x = backend.dot(cell_inputs, self.input_kernel)
        matrix_x = backend.bias_add(matrix_x, input_bias)

        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)

        # hidden state projected by all gate matrices at once
        matrix_h = backend.dot(h_tm1, self.recurrent_kernel)
        matrix_h = backend.bias_add(matrix_h, recurrent_bias)

        recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_h, 3, axis=1)

        z = self.activation(x_z + recurrent_z)
        r = self.activation(x_r + recurrent_r)
        hh = self.recurrent_activation(x_h + r * recurrent_h)

        # previous and candidate state mixed by update gate
        h = h_tm1 + ((1 - z) * h_tm1 + z * hh) * self.dt
        return h, [h]


class Phy_GRUb_cell(layers.AbstractRNNCell):

    def __init__(self,
                 units,
                 activation=tf.sigmoid,
                 recurrent_activation=tf.tanh,
                 kernel_regularizer=regularizer,
                 recurrent_regularizer=regularizer,
                 bias_regularizer=regularizer,
                 dt_regularizer=regularizer,
                 mindt = 1e-4, 
                 maxdt = 1.0,
                 **kwargs):
        self.units = units[0]
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.dt_regularizer = dt_regularizer
        self.mindt = mindt
        self.maxdt = maxdt
        super(Phy_GRUb_cell, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "recurrent_activation": self.recurrent_activation,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "dt_regularizer": self.dt_regularizer,
            "mindt": self.mindt,
            "maxdt": self.maxdt,
        })
        return config

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):

        nfeature = input_shape[-1]

        # kernel:
        self.input_kernel = self.add_weight(shape=(nfeature, 3 * self.units), initializer='uniform',
                                            regularizer=self.kernel_regularizer,
                                            name='input_kernel')
        # recurrent kernel:
        self.recurrent_kernel = self.add_weight(shape=(self.units, 3 * self.units), initializer='uniform',
                                                regularizer=self.recurrent_regularizer,
                                                name='recurrent_kernel')
        # bias:
        self.biases = self.add_weight(shape=(2, 3 * self.units),
                                      initializer=tf.keras.initializers.Zeros(),
                                      regularizer=self.bias_regularizer,
                                      name='biases')
        # delta time:
        self.dt = self.add_weight(shape=(1,), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=self.mindt, max_value=self.maxdt, rate=1.0, axis=0),
                                  regularizer=self.dt_regularizer,
                                  name='dt')

        self.built = True

    def call(self, cell_inputs, cell_states):

        # extract inputs:
        h_tm1 = cell_states[0]
        input_bias, recurrent_bias = tf.unstack(self.biases)

        # inputs projected by all gate matrices at once
        matrix_x = backend.dot(cell_inputs, self.input_kernel)
        matrix_x = backend.bias_add(matrix_x, input_bias)

        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)

        # hidden state projected by all gate matrices at once
        matrix_h = backend.dot(h_tm1, self.recurrent_kernel)
        matrix_h = backend.bias_add(matrix_h, recurrent_bias)

        recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_h, 3, axis=1)

        z = self.activation(x_z + recurrent_z)
        r = self.activation(x_r + recurrent_r)
        hh = self.recurrent_activation(x_h + r * recurrent_h)

        # previous and candidate state mixed by update gate
        h = h_tm1 + hh * self.dt

        return h, [h]


# with density bias caused by pressure change
class ProxyCell(layers.AbstractRNNCell):

    def __init__(self, units, regularizer, **kwargs):
        self.units = units[0]
        self.nprod = units[0]
        self.ninjt = units[1]
        self.regularizer = regularizer
        # print(units)
        super(ProxyCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):
        # print(input_shape)
        self.coeffname = ['Kij', 'ConductRock', 'RockTempBias',
                          'Sink', 'DensityP_W', 'ConductI', 'DensityT_W']

        # coefficients for producer-injector convection
        self.kernelI = self.add_weight(shape=(1, self.nprod, self.ninjt),
                                       initializer='uniform',
                                       constraint=tf.keras.constraints.NonNeg(),
                                       regularizer=self.regularizer,
                                       name='Kij')

        # coefficients for rock conductivity
        self.kr = self.add_weight(shape=(1, 1), initializer=tf.keras.initializers.Zeros(),
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=self.regularizer,
                                  name='ConductRock')

        self.tr = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  # constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=self.regularizer,
                                  name='RockTempBias')

        # coefficients for conduction with injected water
        self.ki = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  regularizer=self.regularizer,
                                  constraint=tf.keras.constraints.NonNeg(), name='ConductI')

        # coefficients for sink term
        self.ksT = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                   constraint=tf.keras.constraints.NonNeg(),
                                   regularizer=self.regularizer,
                                   name='SinkT')

        self.ksP = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                   constraint=tf.keras.constraints.NonNeg(),
                                   regularizer=self.regularizer,
                                   name='SinkP')

        self.ksU = self.add_weight(shape=(self.nprod, self.nprod), initializer='uniform',
                                   # constraint=tf.keras.constraints.NonNeg(),
                                   regularizer=self.regularizer,
                                   name='SinkU')

        # coefficients for density bias
        self.rhoP = self.add_weight(shape=(self.nprod, self.nprod), initializer='uniform',
                                    constraint=tf.keras.constraints.NonNeg(),
                                    regularizer=self.regularizer,
                                    name='DensityP_W')

        # delta time:
        self.dt = self.add_weight(shape=(1,), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=1e-4, max_value=1.0, rate=1.0, axis=0),
                                  regularizer=self.regularizer,
                                  name='dt')

        self.Const1 = tf.cast(np.ones((1, self.ninjt, 1)), dtype=tf.float32)
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        control = layers.Lambda(lambda x: x[:, :self.nprod])(inputs)
        dprodBHP = layers.Lambda(lambda x: x[:, self.nprod:2 * self.nprod])(inputs)
        bhppred = layers.Lambda(lambda x: x[:, 2 * self.nprod:])(inputs)
        ProdBHP = layers.Lambda(lambda x: x[:, :self.nprod])(bhppred)

        Pprod_3D = layers.Lambda(lambda x: x[:, :self.nprod, None])(bhppred)
        Pinjt_3D = layers.Lambda(lambda x: x[:, None, -self.ninjt:])(bhppred)

        # sink term
        self.dTsink = control @ self.ksU * (
                    self.ksT * prev_output + self.ksT * ProdBHP)  # (None x Np) @ (Np x Np) * (None x Np)

        # conduction: Rock and Injection
        self.dTcondR = self.kr * (1 - prev_output + self.tr)
        self.dTcondI = self.ki * (prev_output)

        # convection: Injector
        dP3D = Pinjt_3D - Pprod_3D  # (None x Np x Ni)
        # print(self.kernelI.shape, dP3D.shape)
        dTconv0 = WiseMultiply(self.kernelI, name='ConvectionLayer')(dP3D)
        self.dTconvI = layers.Lambda(lambda x: x[:, :, 0])(tf.matmul(dTconv0, self.Const1))  # reduce_sum: (None x Np)

        # density bias
        self.dTrhoP = dprodBHP @ self.rhoP
        self.dT = (self.dTcondR - self.dTconvI - self.dTsink - self.dTrhoP) * self.dt

        output = prev_output + self.dT
        return output, output


# vanilla RNN
def build_RNN(nfeature, ncontrol, lr=1e-3, regularizer=None):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='history')
    outputs = layers.SimpleRNN(nfeature,
                               return_sequences=True,
                               kernel_regularizer=regularizer,
                               recurrent_regularizer=regularizer,
                               bias_regularizer=regularizer,
                               unroll=False,
                               name='rnn')(control, initial_state=history)
    model = tf.keras.Model(inputs=[control, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt)
    return model


# vanilla GRU
def build_GRU(nfeature, ncontrol, lr=1e-3, regularizer=None):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='history')
    outputs = layers.GRU(nfeature,
                         return_sequences=True,
                         kernel_regularizer=regularizer,
                         recurrent_regularizer=regularizer,
                         bias_regularizer=regularizer,
                         unroll=False,
                         name='gru')(control, initial_state=history)
    model = tf.keras.Model(inputs=[control, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt)
    return model


# phy-RNN
def build_phy_RNN(nfeature, ncontrol, lr=1e-3, regularizer=None, mindt = 1e-4, maxdt = 1.0):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='history')
    pg_rnn = layers.RNN(
        Phy_RNN_cell(
            [nfeature, ncontrol], 
            kernel_regularizer=regularizer, 
            recurrent_regularizer=regularizer,  
            bias_regularizer=regularizer,     
            dt_regularizer=regularizer,      
            mindt = mindt,      
            maxdt = maxdt  
            ),      
            return_sequences=True, unroll=False, name='pg_gru')
    outputs = pg_rnn(control, initial_state=history)
    model = tf.keras.Model(inputs=[control, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt)
    return model


# phy-GRU
def build_phy_GRU(nfeature, ncontrol, lr=1e-3, regularizer=None, mindt = 1e-4, maxdt = 1.0):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='history')
    # outputs = layers.GRU(nfeature, return_sequences=True, unroll=False, name='pg_gru')(control, initial_state=history)
    pg_rnn = layers.RNN(
        Phy_GRU_cell(
            [nfeature, ncontrol],
            kernel_regularizer=regularizer, 
            recurrent_regularizer=regularizer,  
            bias_regularizer=regularizer,     
            dt_regularizer=None,      
            mindt = mindt,      
            maxdt = maxdt
            ),
            return_sequences=True, unroll=False, name='pg_gru')
    outputs = pg_rnn(control, initial_state=history)
    model = tf.keras.Model(inputs=[control, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt)
    return model


# Two-Layer GRU: GRU + GRU
def build_GRU2(nfeature, ncontrol, nz=3, lr=1e-3, regularizer=None, mindt = 1e-4, maxdt = 1.0):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='history')
    # outputs = layers.GRU(nfeature, return_sequences=True, unroll=False, name='pg_gru')(control, initial_state=history)
    z = layers.GRU(nz, return_sequences=True, unroll=False, name='gru0')(control)

    outputs = layers.GRU(nfeature, return_sequences=True, unroll=False, name='gru1')(z, initial_state=history)
#     outputs = pg_rnn(z, initial_state=history)
    model = tf.keras.Model(inputs=[control, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt)
    return model


# Two-Layer phy-RNN: GRU + phy-RNN
def build_phy_RNN2(nfeature, ncontrol, nz=3, lr=1e-3, regularizer=None, mindt = 1e-4, maxdt = 1.0):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='history')
    # outputs = layers.GRU(nfeature, return_sequences=True, unroll=False, name='pg_gru')(control, initial_state=history)
    z = layers.GRU(nz, return_sequences=True, unroll=False, name='gru0')(control)
    pg_rnn = layers.RNN(
        Phy_RNN_cell(
            [nfeature, ncontrol],
            kernel_regularizer=regularizer, 
            recurrent_regularizer=regularizer,  
            bias_regularizer=regularizer,     
            dt_regularizer=regularizer,      
            mindt = mindt,      
            maxdt = maxdt
            ),
            return_sequences=True, unroll=False, name='pg_rnn')
    outputs = pg_rnn(z, initial_state=history)
    model = tf.keras.Model(inputs=[control, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt)
    return model



# Two-Layer phy-GRU: GRU + phy-GRU
def build_phy_GRU2(nfeature, ncontrol, nz=3, lr=1e-3, regularizer=None, mindt = 1e-4, maxdt = 1.0):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='history')
    # outputs = layers.GRU(nfeature, return_sequences=True, unroll=False, name='pg_gru')(control, initial_state=history)
    z = layers.GRU(nz, return_sequences=True, unroll=False, name='gru0')(control)
    pg_rnn = layers.RNN(
        Phy_GRU_cell(
            [nfeature, ncontrol],
            kernel_regularizer=regularizer, 
            recurrent_regularizer=regularizer,  
            bias_regularizer=regularizer,     
            dt_regularizer=regularizer,      
            mindt = mindt,      
            maxdt = maxdt
            ),
            return_sequences=True, unroll=False, name='pg_gru')
    outputs = pg_rnn(z, initial_state=history)
    model = tf.keras.Model(inputs=[control, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt)
    return model


# labeling scheme with regularized phy-GRU
def build_phy_RNN_L(nfeature, ncontrol, lr=1e-3, regularizer=None, mindt = 1e-4, maxdt = 1.0, decay_steps=1000):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='History')
    labels = tf.keras.Input(batch_shape=(None, None, nfeature), name='Labels')
    # outputs = layers.GRU(nfeature, return_sequences=True, unroll=False, name='pg_gru')(control, initial_state=history)
    pg_rnn = layers.RNN(
        Phy_RNN_cell(
            [nfeature, ncontrol],
            kernel_regularizer=regularizer, 
            recurrent_regularizer=regularizer,  
            bias_regularizer=regularizer,     
            dt_regularizer=regularizer,      
            mindt = mindt,      
            maxdt = maxdt
            ),
            return_sequences=True, unroll=False, name='pg_rnn')
    outputs = pg_rnn(control, initial_state=history)
    moutputs = layers.Multiply()([labels, outputs])
    model = tf.keras.Model(inputs=[control, history, labels], outputs=[moutputs, outputs])
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=decay_steps, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mse'], loss_weights=[1, 0], optimizer=opt)
    return model


# labeling scheme with regularized phy-GRU
def build_phy_GRU_L(nfeature, ncontrol, lr=1e-3, regularizer=None, mindt = 1e-4, maxdt = 1.0, decay_steps=1000):
    control = tf.keras.Input(batch_shape=(None, None, ncontrol), name='Control')
    history = tf.keras.Input(batch_shape=(None, nfeature), name='History')
    labels = tf.keras.Input(batch_shape=(None, None, nfeature), name='Labels')
    # outputs = layers.GRU(nfeature, return_sequences=True, unroll=False, name='pg_gru')(control, initial_state=history)
    pg_rnn = layers.RNN(
        Phy_GRU_cell(
            [nfeature, ncontrol],
            kernel_regularizer=regularizer, 
            recurrent_regularizer=regularizer,  
            bias_regularizer=regularizer,     
            dt_regularizer=None,      
            mindt = mindt,      
            maxdt = maxdt
            ),
            return_sequences=True, unroll=False, name='pg_gru')
    outputs = pg_rnn(control, initial_state=history)
    moutputs = layers.Multiply()([labels, outputs])
    model = tf.keras.Model(inputs=[control, history, labels], outputs=[moutputs, outputs])
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=decay_steps, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=['mse', 'mse'], loss_weights=[1, 0], optimizer=opt)
    return model



def ProxyTempLayer(nprod, ninjt, lr=1e-3):
    predBHP = tf.keras.Input(batch_shape=(None, None, nprod + ninjt), name='BHP')
    dprodBHP= tf.keras.Input(batch_shape=(None, None, nprod), name='dPBHP')
    controlP = tf.keras.Input(batch_shape=(None, None, nprod), name='Control')
    initialT = tf.keras.Input(batch_shape=(None, nprod), name='initialT')
    inputs = layers.Concatenate(axis=-1)([controlP, dprodBHP, predBHP])
    pg_rnn = layers.RNN(ProxyCell([nprod, ninjt]), return_sequences=True, unroll=False, name='pg_rnn')
    outputs = pg_rnn(inputs, initial_state=initialT)
    model = tf.keras.Model(inputs=[controlP, dprodBHP, predBHP, initialT], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt, metrics=['accuracy'])
    return model


# use NN to predict pressure
def build_ProxyNN(nprod, ninjt, lr, regularizer=None, decay_steps=1000):
    initialT = tf.keras.Input(batch_shape=(None, nprod), name='T0') # temperature of step 0
    Control0 = tf.keras.Input(batch_shape=(None, 1, nprod+ninjt), name='x0') # only BHP
    controlT = tf.keras.Input(batch_shape=(None, None, nprod+ninjt), name='Control') # production + injection rate
    controlP = layers.Lambda(lambda x: x[:,:,:nprod])(controlT) # production rate
    labels = tf.keras.Input(batch_shape=(None, None, nprod), name='Labels')
    
    controlT_ = layers.Concatenate(axis=1)([Control0, controlT]) 
    model_T = ProxyTempLayer(nprod, ninjt, lr=lr)
    
    NN = layers.Dense(nprod + ninjt, activation='tanh', use_bias=True, 
                      kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                      kernel_regularizer=regularizer, bias_regularizer=regularizer)
    predBHP_ = layers.TimeDistributed(NN)(controlT_)
    predBHP  = layers.Lambda(lambda x: x[:,1:,:])(predBHP_)
    
    dbhp_pred = layers.Lambda(lambda x: x[:,1:,:]-x[:,:-1,:])(predBHP_) # differential pressure
    dprodBHP  = layers.Lambda(lambda x: x[:,:,:nprod])(dbhp_pred) # differential production pressure
    
    # predict temperature
    outputs = model_T([controlP, dprodBHP, predBHP, initialT])
    moutputs = layers.Multiply()([labels, outputs])
    
    # compile model
    model = tf.keras.Model(inputs=[Control0, initialT, controlT, labels], outputs=[moutputs, outputs])
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=decay_steps, decay_rate=0.95, staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=optimizer)
    return model
