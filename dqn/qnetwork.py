
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras.layers import TimeDistributed, LSTM, GRU
from keras.layers import add, dot
from keras.models import Model
from keras import backend as K


def qnetwork_add_arguments(parser):
    parser.add_argument('--model_name', default='dqn', type=str,
        help='Model name')
    parser.add_argument('--dense_size', default=256, type=int,
        help='Number of hidden units in the dense layer')

def create_model(height, width, num_frames, num_act, args):
    # input state
    state = Input(shape=(height, width, num_frames))

    # convolutional layers
    conv1_16 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')
    conv2_32 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')
    conv1_32 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

    # if recurrent net then change input shape
    if 'drqn' in args.model_name:
        # recurrent net (dqn)
        state_shape_drqn = num_frames, height, width, 1
        lambda_perm_state = lambda x: K.permute_dimensions(x, [0, 3, 1, 2])
        perm_state = Lambda(lambda_perm_state)(state)
        dist_state = Lambda(lambda x: K.stack([x], axis=4))(perm_state)

        # extract feature with convolutional layers
        if 'cheap' in args.model_name:
            dist_conv1 = TimeDistributed(conv1_16)(dist_state)
            dist_convf = TimeDistributed(conv2_32)(dist_conv1)
        else:
            dist_conv1 = TimeDistributed(conv1_32)(dist_state)
            dist_conv2 = TimeDistributed(conv2_64)(dist_conv1)
            dist_convf = TimeDistributed(conv3_64)(dist_conv2)
        feature = TimeDistributed(Flatten())(dist_convf)

        # dueling or regular drqn
        if 'dueling' in args.model_name:
            if 'val_lstm' in args.model_name:
                val_type = LSTM
            elif 'val_gru' in args.model_name:
                val_type = GRU
            if 'adv_lstm' in args.model_name:
                adv_type = LSTM
            elif 'adv_gru' in args.model_name:
                adv_type = GRU
            value1 = val_type(args.dense_size, activation='relu')(feature)
            adv1 = adv_type(args.dense_size, activation='relu')(feature)
            q_value = dueling(num_act, value1, adv1)
        else:
            if 'lstm' in args.model_name:
                net_type = LSTM
            elif 'gru' in args.model_name:
                net_type = GRU
            hid = net_type(args.dense_size, activation='relu')(feature)
            q_value = Dense(num_act)(hid)
    elif 'dqn' in args.model_name:
        # fully connected net (dqn)
        # extract feature with convolutional layers
        if 'cheap' in args.model_name:
            conv1 = conv1_16(state)
            convf = conv2_32(conv1)
        else:
            conv1 = conv1_32(state)
            conv2 = conv2_64(conv1)
            convf = conv3_64(conv2)
        feature = Flatten()(convf)

        # dueling or regular dqn
        if 'dueling' in args.model_name:
            value1 = Dense(args.dense_size, activation='relu')(feature)
            adv1 = Dense(args.dense_size, activation='relu')(feature)
            q_value = dueling(num_act, value1, adv1)
        else:
            hid = Dense(args.dense_size, activation='relu')(feature)
            q_value = Dense(num_act)(hid)

    # build model
    act = Input(shape=(num_act,))
    q_value_act = dot([q_value, act], axes=1)
    model = Model(inputs=[state, act], outputs=[q_value_act, q_value])
    return model

def dueling(num_act, value1, adv1):
    value2 = Dense(1)(value1)
    adv2 = Dense(num_act)(adv1)
    mean_adv2 = Lambda(lambda x: K.mean(x, axis=1))(adv2)
    ones = K.ones([1, num_act])
    lambda_exp = lambda x: K.dot(K.expand_dims(x, axis=1), -ones)
    exp_mean_adv2 = Lambda(lambda_exp)(mean_adv2)
    sum_adv = add([exp_mean_adv2, adv2])
    exp_value2 = Lambda(lambda x: K.dot(x, ones))(value2)
    q_value = add([exp_value2, sum_adv])
    return q_value

