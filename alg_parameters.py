""" Hyperparameters"""


class EnvParameters:
    N_ACTIONS = 5
    FOV_SIZE = 9
    NUM_TIME_SLICE=9
    WINDOWS=15
    UTI_WINDOWS=(-2,16)
    DIS_TIME_WEIGHT = (0.9, 0.1)
    K_STEPS=5


class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 31  # number of channels of observations
    GOAL_REPR_SIZE = 32
    VECTOR_LEN = 8
    GAIN=0.01
    DY_HIDDEN = 64
    PAST_HIDDEN= 64
    TIME_DEPT=4


class SetupParameters:
    SEED = 1234




