
from keras.layers import Conv2D, Dense, Flatten, Input, Lambda, add, dot
from keras import backend as K
from keras.models import Model


def create_model(state_shape, num_act, model_name):
    state = Input(shape=state_shape)
    if 'cheap' in model_name:
        conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(state)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
        feature = Flatten()(conv2)
        if model_name == 'cheap_dqn':
            hid = Dense(128, activation='relu')(feature)
            q_value = Dense(num_act)(hid)
        elif model_name == 'cheap_dueling_dqn':
            value1 = Dense(128, activation='relu')(feature)
            value2 = Dense(1)(value1)
            adv1 = Dense(128, activation='relu')(feature)
            adv2 = Dense(num_act)(adv1)
            mean_adv2 = Lambda(lambda x: K.mean(x, axis=1))(adv2)
            ones = K.ones([1, num_act])
            exp_mean_adv2 = Lambda(lambda x: K.dot(K.expand_dims(x, axis=1), -ones))(mean_adv2)
            sum_adv = add([exp_mean_adv2, adv2])
            exp_value2 = Lambda(lambda x: K.dot(x, ones))(value2)
            q_value = add([exp_value2, sum_adv])
    else:
        conv1 = Conv2D(32, (8, 8), strides=(4, 4),
            padding='same', activation='relu')(state)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2),
            padding='same', activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1),
            padding='same', activation='relu')(conv2)
        feature = Flatten()(conv3)
        if model_name == 'dqn':
            hid = Dense(512, activation='relu')(feature)
            q_value = Dense(num_act)(hid)
        elif model_name == 'dueling_dqn':
            value1 = Dense(512, activation='relu')(feature)
            value2 = Dense(1)(value1)
            advantage1 = Dense(512, activation='relu')(feature)
            advantage2 = Dense(num_act)(advantage1)
            mean_advantage2 = Lambda(lambda x: K.mean(x, axis=1))(advantage2)
            ones = K.ones([1, num_act])
            exp_mean_advantage2 = Lambda(lambda x: K.dot(K.expand_dims(x, axis=1), -ones))(mean_advantage2)
            sum_adv = add([exp_mean_advantage2, advantage2])
            exp_value2 = Lambda(lambda x: K.dot(x, ones))(value2)
            q_value = add([exp_value2, sum_adv])
    act = Input(shape=(num_act,))
    q_value_act = dot([q_value, act], axes=1)
    model = Model(inputs=[state, act], outputs=[q_value_act, q_value])
    return model
