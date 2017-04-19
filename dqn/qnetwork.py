
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras.layers import add, dot
from keras.models import Model
from keras import backend as K


def qnetwork_add_arguments(parser):
    parser.add_argument('--model_name', default='dqn', type=str,
        help='Model name')
    parser.add_argument('--dense_size', default=256, type=int,
        help='Number of hidden units in the dense layer')

def create_model(state_shape, num_act, args):
    # input state
    state = Input(shape=state_shape)

    # extract feature with convolutional layers
    if 'cheap' in args.model_name:
        conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(state)
        conv_f = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
    else:
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(state)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv_f = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    feature = Flatten()(conv_f)

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

