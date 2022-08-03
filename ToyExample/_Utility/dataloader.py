import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def GenerateSets(x, y, frequency, twindow, twindow2, predictwindow):
    yin = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow, y.shape[1]))
    xin = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow, x.shape[1]))
    yout = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow2, y.shape[1]))
    control = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow2, x.shape[1]))
    obs = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, y.shape[1]))
    for i in range(x.shape[0] - (twindow + twindow2) * frequency + 1):
        for j in range(twindow):
            yin[i, j] = y[i + j * frequency]
            xin[i, j] = x[i + j * frequency]
        for j in range(twindow2):
            yout[i, j] = y[twindow * frequency + i + j * frequency]
            control[i, j] = x[twindow * frequency + i + j * frequency]
        obs[i] = y[twindow * frequency + i]
    history = np.concatenate((xin, yin), axis=2)
    n = int(yout.shape[0] / twindow2 / frequency)
    traintesthistory = np.zeros((n, twindow, x.shape[1] + y.shape[1]))
    traintestcontrol = np.zeros((n, twindow2, x.shape[1]))
    for i in range(n):
        traintesthistory[i] = history[i * twindow2 * frequency]
        traintestcontrol[i] = control[i * twindow2 * frequency]
    stepsize = 1
    initialwindow = yout.shape[0] - predictwindow
    index = np.random.permutation(initialwindow)
    historyt = (history[:initialwindow])[index]
    controlt = (control[:initialwindow])[index]
    youtt = (yout[:initialwindow])[index]
    return history, control, yout, historyt, controlt, youtt, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin


def normalization(prop):
    # prop = block_reduce(prop, block_size = (24,1), func = np.mean)
    # prop = scipy.signal.savgol_filter(prop[:-1], 1001, 2, mode = 'mirror',axis = 0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return np.reshape(scaler.fit_transform(prop.reshape([-1, 1])), [-1, prop.shape[1]])


def load_samples(path):
    ENTH_sim = sio.loadmat(path)['ENTH']
    PTAV_sim = sio.loadmat(path)['PTAV']
    PPAV_sim = sio.loadmat(path)['PPAV']
    PBHP_sim = sio.loadmat(path)['PBHP']
    IBHP_sim = sio.loadmat(path)['IBHP']
    RATE_sim = sio.loadmat(path)['RATE']
    RATEi_sim= sio.loadmat(path)['RATEi']
    return ENTH_sim, PTAV_sim, PPAV_sim, PBHP_sim, IBHP_sim, RATE_sim, RATEi_sim