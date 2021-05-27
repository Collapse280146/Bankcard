from keras import backend as K

def ctc_loss_layer(args):
    y_true, y_pred, pred_length, label_length = args
    batch_cost = K.ctc_batch_cost(y_true, y_pred, pred_length, label_length)
    return batch_cost


def fake_ctc_loss(y_true, y_pred):
    return y_pred


