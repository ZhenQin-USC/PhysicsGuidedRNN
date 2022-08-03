import sys
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


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


class DiagonalWeight(tf.keras.constraints.Constraint):
    """Constrains the weights to be diagonal.
    """

    def __call__(self, w):
        N = tf.keras.backend.int_shape(w)[-1]
        m = tf.keras.backend.eye(N)
        w *= m
        return w


class NonNegDiagonal(tf.keras.constraints.Constraint):
    """Constrains the weights to be non-negative diagonal.
    """

    def __call__(self, w):
        N = tf.keras.backend.int_shape(w)[-1]
        m = tf.keras.backend.eye(N)
        w *= tf.cast(tf.math.greater_equal(w, 0.), w.dtype)
        w *= m
        return w


class ProxyCell(layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units[0]
        self.nprod = units[0]
        self.ninjt = units[1]
        super(ProxyCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):
        self.coeffname = ['Kij', 'Kii', 'ConductRock', 'RockTempBias', 'SinkTempBias',
                          'Sink', 'RechargePressure', 'RechargeContribution', 'ConductI']
        # coefficients for producer-injector convection
        self.kernelI = self.add_weight(shape=(1, self.nprod, self.ninjt),
                                       initializer='uniform',
                                       constraint=tf.keras.constraints.NonNeg(),
                                       regularizer=tf.keras.regularizers.l1_l2(l1=0.000000, l2=0.00005),
                                       name='Kij')

        # coefficients for producers' convection
        #         self.kernelP = self.add_weight(shape=(1, self.nprod, self.nprod),
        #                                        initializer='uniform',
        #                                        #constraint=tf.keras.constraints.NonNeg(),
        #                                        regularizer=tf.keras.regularizers.l1_l2(l1=0.000000, l2=0.00005),
        #                                        name='Kii')

        # coefficients for rock conductivity
        self.kr = self.add_weight(shape=(1, 1), initializer=tf.keras.initializers.Zeros(),
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='ConductRock')
        self.tr = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  # constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='RockTempBias')

        self.ti = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  # constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='ConductionLossBias')
        # coefficients for sink term
        #         self.ts = self.add_weight(shape=(1,self.nprod), initializer='uniform',
        #                                   #constraint=tf.keras.constraints.NonNeg(),
        #                                   regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
        #                                   name='SinkTempBias')
        self.ks = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
                                  name='Sink')

        # coefficients for recharge
        #         self.pre = self.add_weight(shape=(1,self.nprod), initializer='uniform',
        #                                    #constraint=tf.keras.constraints.NonNeg(),
        #                                    regularizer=tf.keras.regularizers.l1_l2(l1=0.00001, l2=0.00005),
        #                                    name='RechargePressureBias')
        #         self.cre = self.add_weight(shape=(1,self.nprod), initializer='uniform',
        #                                    constraint=tf.keras.constraints.NonNeg(),
        #                                    regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
        #                                    name='RechargeContribution')

        # coefficients for conduction with injected water
        self.ki = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                  constraint=tf.keras.constraints.NonNeg(), name='ConductI')


        # delta time:
        self.dt = self.add_weight(shape=(1,), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=1e-4, max_value=1.0, rate=1.0, axis=0),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000001, l2=0.000001),
                                  name='dt')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        control = layers.Lambda(lambda x: x[:, :self.nprod])(inputs)
        bhppred = layers.Lambda(lambda x: x[:, self.nprod:])(inputs)
        self.Const1 = tf.cast(np.ones((1, self.ninjt, 1)), dtype=tf.float32)
        self.Const2 = tf.cast(np.ones((1, self.nprod, 1)), dtype=tf.float32)
        Pprod = layers.Lambda(lambda x: x[:, :self.nprod, None])(bhppred)
        Pinjt = layers.Lambda(lambda x: x[:, None, -self.ninjt:])(bhppred)

        # sink term
        self.dTsink = self.ks * control * prev_output
        # self.dTsink = tf.keras.activations.tanh((control@self.Ws+self.bs)*prev_output)

        # conduction: Rock and Injection
        self.dTcondR = self.kr * (1 - prev_output + self.tr)
        self.dTcondI = self.ki * (prev_output + self.ti)

        # convection: Injector
        dP = Pinjt - Pprod  # (None x Np x Ni)
        dTconv0 = WiseMultiply(self.kernelI, name='ConvectionLayer')(dP)
        # dTconv1 = layers.Activation(tf.keras.activations.tanh)(dTconv0)
        self.dTconvI = layers.Lambda(lambda x: x[:, :, 0])(tf.matmul(dTconv0, self.Const1))  # reduce_sum: (None x Np)

        # convection: Producer
        #         dPprod = layers.ReLU()(tf.transpose(Pprod, perm=(0,2,1))-Pprod) # (None x Np x Np) row - producer
        #         dTconvP0 = self.kernelP*dPprod
        #         self.dTconvP = layers.Lambda(lambda x: x[:,:,0])(tf.matmul(dTconvP0, self.Const2))

        # print(self.dTconvP.shape)
        #         dTProd = -(layers.Lambda(lambda x: x[:,:,None])(prev_output) -
        #                    layers.Lambda(lambda x: tf.transpose(x[:,:,None], perm=(0,2,1)))(prev_output)) # (None x repeats x Np) column - producer
        #         dTconvP0 = self.kernelP*dPprod*dTProd
        #         self.dTconvP = layers.Lambda(lambda x: x[:,:,0])(tf.matmul(dTconvP0, self.Const2))

        # convection: recharge source
        #         dP_re = layers.Lambda(lambda x: x[:,:,0])(1-Pprod)
        #         self.dTsource = self.cre*dP_re
        self.dT = (self.dTcondR - self.dTconvI - self.dTsink - self.dTcondI) * self.dt

        output = prev_output + self.dT
        return output, output

    def return_coeffname(self):
        return self.coeffname


# with density bias caused pressure change
class ProxyCell2(layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units[0]
        self.nprod = units[0]
        self.ninjt = units[1]
        super(ProxyCell2, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):
        self.coeffname = ['Kij', 'ConductRock', 'RockTempBias',
                          'Sink', 'DensityP_W', 'ConductI', 'DensityT_W']
        # coefficients for producer-injector convection
        self.kernelI = self.add_weight(shape=(1, self.nprod, self.ninjt),
                                       initializer='uniform',
                                       constraint=tf.keras.constraints.NonNeg(),
                                       regularizer=tf.keras.regularizers.l1_l2(l1=0.000000, l2=0.00005),
                                       name='Kij')

        # coefficients for rock conductivity
        self.kr = self.add_weight(shape=(1, 1), initializer=tf.keras.initializers.Zeros(),
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='ConductRock')
        self.tr = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  # constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='RockTempBias')

        # coefficients for sink term
        self.ks = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
                                  name='Sink')

        # coefficients for density bias
        self.rhoP = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                    constraint=tf.keras.constraints.NonNeg(),
                                    regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                    name='DensityP_W')

        # coefficients for conduction with injected water
        self.ki = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                  constraint=tf.keras.constraints.NonNeg(), name='ConductI')

        # delta time:
        self.dt = self.add_weight(shape=(1,), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=1e-4, max_value=1.0, rate=1.0, axis=0),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000001, l2=0.000001),
                                  name='dt')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        pcontrol = layers.Lambda(lambda x: x[:, :self.nprod])(inputs)
        dprodBHP= layers.Lambda(lambda x: x[:, self.nprod:2*self.nprod])(inputs)
        bhppred = layers.Lambda(lambda x: x[:, 2*self.nprod:])(inputs)
        self.Const1 = tf.cast(np.ones((1, self.ninjt, 1)), dtype=tf.float32)
        self.Const2 = tf.cast(np.ones((1, self.nprod, 1)), dtype=tf.float32)
        Pprod = layers.Lambda(lambda x: x[:, :self.nprod, None])(bhppred)
        Pinjt = layers.Lambda(lambda x: x[:, None, -self.ninjt:])(bhppred)

        # sink term
        self.dTsink = self.ks * pcontrol * prev_output

        # conduction: Rock and Injection
        self.dTcondR = self.kr * (1 - prev_output + self.tr)
        self.dTcondI = self.ki * (prev_output + 0.3)

        # convection: Injector
        dP = Pinjt - Pprod  # (None x Np x Ni)
        dTconv0 = WiseMultiply(self.kernelI, name='ConvectionLayer')(dP)
        self.dTconvI = layers.Lambda(lambda x: x[:, :, 0])(tf.matmul(dTconv0, self.Const1))  # reduce_sum: (None x Np)

        # density bias
        self.dTrhoP = self.rhoP*dprodBHP

        self.dT = (self.dTcondR - self.dTconvI - self.dTcondI - self.dTsink - self.dTrhoP) * self.dt
        output = prev_output + self.dT
        # output = prev_output - self.dTconvI - self.dTcondI + self.dTcondR - self.dTsink - self.dTrhoP
        return output, output

# with density bias caused by both P and T
class ProxyCell3(layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units[0]
        self.nprod = units[0]
        self.ninjt = units[1]
        super(ProxyCell3, self).__init__(**kwargs)

    @property
    def state_size(self):
        return self.units  # (None, self.units)

    def build(self, input_shape):
        self.coeffname = ['Kij', 'ConductRock', 'RockTempBias', 'Sink', 'DensityP_W', 'DensityT_W', 'DensityT_Bias', 'ConductI']
        # coefficients for producer-injector convection
        self.kernelI = self.add_weight(shape=(1, self.nprod, self.ninjt),
                                       initializer='uniform',
                                       constraint=tf.keras.constraints.NonNeg(),
                                       regularizer=tf.keras.regularizers.l1_l2(l1=0.000000, l2=0.00005),
                                       name='Kij')

        # coefficients for rock conductivity
        self.kr = self.add_weight(shape=(1, 1), initializer=tf.keras.initializers.Zeros(),
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='ConductRock')
        self.tr = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  # constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='RockTempBias')

        # coefficients for sink term
        self.ks = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
                                  name='Sink')

        # coefficients for density bias caused by P
        self.rhoP_W = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                      name='DensityP_W')

        # coefficients for density bias caused by T
        self.rhoT_W = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                      name='DensityT_W')
        self.rhoT_b = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                      constraint=tf.keras.constraints.NonNeg(),
                                      regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                      name='DensityT_Bias')
        # coefficients for conduction with injected water
        self.ki = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                  constraint=tf.keras.constraints.NonNeg(), name='ConductI')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        control = layers.Lambda(lambda x: x[:, :self.nprod])(inputs)
        dprodBHP= layers.Lambda(lambda x: x[:, self.nprod:2*self.nprod])(inputs)
        bhppred = layers.Lambda(lambda x: x[:, 2*self.nprod:])(inputs)
        self.Const1 = tf.cast(np.ones((1, self.ninjt, 1)), dtype=tf.float32)
        self.Const2 = tf.cast(np.ones((1, self.nprod, 1)), dtype=tf.float32)
        Pprod = layers.Lambda(lambda x: x[:, :self.nprod, None])(bhppred)
        Pinjt = layers.Lambda(lambda x: x[:, None, -self.ninjt:])(bhppred)

        # sink term
        self.dTsink = self.ks * control * prev_output

        # conduction: Rock and Injection
        self.dTcondR = self.kr * (1 - prev_output + self.tr)
        self.dTcondI = self.ki * (prev_output + 0.3)

        # convection: Injector
        dP = Pinjt - Pprod  # (None x Np x Ni)
        dTconv0 = WiseMultiply(self.kernelI, name='ConvectionLayer')(dP)
        self.dTconvI = layers.Lambda(lambda x: x[:, :, 0])(tf.matmul(dTconv0, self.Const1))  # reduce_sum: (None x Np)

        # density bias
        self.dTrhoP = self.rhoP_W*dprodBHP
        self.rho_bias_T = self.rhoT_b-self.rhoT_W*prev_output

        # temperature decline
        self.dT = (self.dTcondR - self.dTconvI - self.dTcondI - self.dTsink - self.dTrhoP)*self.rho_bias_T
        output = prev_output + self.dT
        return output, output


# with density bias caused by pressure change
class ProxyCell4(layers.AbstractRNNCell):

    def __init__(self, units, **kwargs):
        self.units = units[0]
        self.nprod = units[0]
        self.ninjt = units[1]
        # print(units)
        super(ProxyCell4, self).__init__(**kwargs)

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
                                       regularizer=tf.keras.regularizers.l1_l2(l1=0.000000, l2=0.00005),
                                       name='Kij')

        # coefficients for rock conductivity
        self.kr = self.add_weight(shape=(1, 1), initializer=tf.keras.initializers.Zeros(),
                                  constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='ConductRock')

        self.tr = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  # constraint=tf.keras.constraints.NonNeg(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000000, l2=0.00005),
                                  name='RockTempBias')

        # coefficients for conduction with injected water
        self.ki = self.add_weight(shape=(1, self.nprod), initializer=tf.keras.initializers.Zeros(),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                  constraint=tf.keras.constraints.NonNeg(), name='ConductI')

        # coefficients for sink term
        self.ksT = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                   constraint=tf.keras.constraints.NonNeg(),
                                   regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
                                   name='SinkT')

        self.ksP = self.add_weight(shape=(1, self.nprod), initializer='uniform',
                                   constraint=tf.keras.constraints.NonNeg(),
                                   regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
                                   name='SinkP')

        self.ksU = self.add_weight(shape=(self.nprod, self.nprod), initializer='uniform',
                                   # constraint=tf.keras.constraints.NonNeg(),
                                   regularizer=tf.keras.regularizers.l1_l2(l1=0.000005, l2=0.00005),
                                   name='SinkU')

        # coefficients for density bias
        self.rhoP = self.add_weight(shape=(self.nprod, self.nprod), initializer='uniform',
                                    constraint=tf.keras.constraints.NonNeg(),
                                    regularizer=tf.keras.regularizers.l1_l2(l1=0.000001, l2=0.00005),
                                    name='DensityP_W')

        # delta time:
        self.dt = self.add_weight(shape=(1,), initializer='uniform',
                                  constraint=tf.keras.constraints.MinMaxNorm(
                                      min_value=1e-4, max_value=1.0, rate=1.0, axis=0),
                                  regularizer=tf.keras.regularizers.l1_l2(l1=0.0000001, l2=0.000001),
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


def build_PGRNN(nprod, ninjt, lr=1e-3):
    predBHP = tf.keras.Input(batch_shape=(None, None, nprod + ninjt), name='BHP')
    control = tf.keras.Input(batch_shape=(None, None, nprod), name='Control')
    history = tf.keras.Input(batch_shape=(None, nprod), name='History')
    inputs = layers.Concatenate(axis=-1)([control, predBHP])
    pg_rnn = layers.RNN(ProxyCell([nprod, ninjt]), return_sequences=True, unroll=False, name='pg_rnn')
    outputs = pg_rnn(inputs, initial_state=history)
    model = tf.keras.Model(inputs=[control, predBHP, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt, metrics=['accuracy'])
    return model


def build_Proxy(nprod, ninjt, lr=1e-3):
    predBHP = tf.keras.Input(batch_shape=(None, None, nprod + ninjt), name='BHP')
    dprodBHP= tf.keras.Input(batch_shape=(None, None, nprod), name='dPBHP')
    control = tf.keras.Input(batch_shape=(None, None, nprod), name='Control')
    history = tf.keras.Input(batch_shape=(None, nprod), name='History')
    inputs = layers.Concatenate(axis=-1)([control, dprodBHP, predBHP])
    pg_rnn = layers.RNN(ProxyCell2([nprod, ninjt]), return_sequences=True, unroll=False, name='pg_rnn')
    outputs = pg_rnn(inputs, initial_state=history)
    model = tf.keras.Model(inputs=[control, dprodBHP, predBHP, history], outputs=outputs)
    initial_learning_rate = lr
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=300, decay_rate=0.95, staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', loss_weights=1, optimizer=opt, metrics=['accuracy'])
    return model

