import sys
sys.path.append(r'D:\Users\qinzh\Google Drive USC\MATLAB Local\Proxy Opt')
from Function.build_model import CustomCallback
import numpy as np
from os.path import join


def train_model(build_model, xtrain, ytrain, x_val, y_val, x_test, nmodel, path_to_data, weight_item,
                nbsize, nepochs, verbose, shuffle, lossbar, loss_bar, valloss_bar):
    k = 1  # record for the number of accepted models
    kk = 1  # record for the number of training trials
    while k <= nmodel:
        print('\n=============Starting the training process No.{}============='.format(kk))
        kk += 1
        # build model
        model_long = build_model([])
        try:
            h = model_long.fit(xtrain, ytrain,
                               validation_data=[x_val, y_val],
                               batch_size=nbsize, epochs=nepochs,
                               verbose=verbose, shuffle=shuffle,
                               callbacks=[CustomCallback(nepochs, lossbar, loss_bar, valloss_bar)])
        except ValueError:
            print('\nRe-initialize the model')
            continue
        except KeyboardInterrupt:
            print('\nManually Interrupt')
            break

        ypred = model_long.predict(x_test)

        matdict = {}
        for ii in range(len(weight_item)):
            matdict[weight_item[ii]] = model_long.get_weights()[ii]

        np.save(join(path_to_data, "weights_trial{}".format(k)), matdict)
        np.save(join(path_to_data, 'loss_trial{}.npy'.format(k)), h.history['loss'])
        np.save(join(path_to_data, 'valoss_trial{}.npy'.format(k)), h.history['val_loss'])
        np.save(join(path_to_data, 'ypred_trial{}.npy'.format(k)), ypred)
        k += 1