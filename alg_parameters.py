import datetime

""" Hyperparameters"""


class EnvParameters:
    LOCAL_N_AGENTS_LIST = [8,8,8]  # number of agents used in training
    GLOBAL_N_AGENT_LIST=[(0.4,0.45,0.5),(0.5,0.55,0.6),(0.6,0.65,0.7)]
    N_ACTIONS = 5
    EPISODE_LEN = [356,356,356]
    FOV_SIZE = 9
    WORLD_SIZE_LIST = [10,25,50]
    OBSTACLE_PROB_LIST = [(0.05,0.075,0.1),(0.1,0.125,0.15),(0.15,0.175,0.2)]
    ACTION_COST = [-0.4,-0.5,-0.6]
    IDLE_COST = [-0.4,-0.5,-0.6]
    ADD_COST=-0.2
    MOVE_BACK_COST=-0.4
    GOAL_REWARD = 0.0
    DY_COLLISION_COST = -1.5
    AG_COLLISION_COST = -1.5
    NUM_TIME_SLICE=9
    WINDOWS=15
    OFF_ROUTE_FACTOR=[0.06,0.05,0.04]
    DIS_FACTOR = 0.2
    SWITCH_TIMESTEP=[1e7,2e7]
    UTI_WINDOWS=list(range(-2,16))
    UTI_WEIGHT=[0.25,0.75] # edge,vertex
    OVERALL_WEIGHT= -0.3
    DIS_TIME_WEIGHT = [0.9, 0.1]
    K_STEPS=5


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 50
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    POLICY_COEF = 1
    VALID_COEF = 0.5
    N_EPOCHS = 10
    N_ENVS = 32 # number of processes
    N_MAX_STEPS = 7e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 8  # number of time steps per process per data collection
    MINIBATCH_SIZE =int(2**9)
    ITERATION_LIMIT_LIST=[30,65,100]
    opti_eps=1e-5
    weight_decay=0
    DEMONSTRATION_THRES=[3e6,2e6,0]

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
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = False
    WANDB = True
    ENTITY = 'your_name'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'MAPF'
    EXPERIMENT_NAME = 'LNS2+RL'
    EXPERIMENT_NOTE = ''
    SAVE_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS*400  # interval of saving model
    PRINT_INTERVAL=5e4
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    LOSS_NAME = ['all_loss', 'policy_loss', 'policy_entropy', 'critic_loss', 'valid_loss','clipfrac',
                 'grad_norm', 'advantage',"prop_policy","prop_en","prop_v","prop_valid"]

all_args = {'LOCAL_N_AGENTS': EnvParameters.LOCAL_N_AGENTS_LIST, "GLOBAL_N_AGENT":EnvParameters.GLOBAL_N_AGENT_LIST,
            'N_ACTIONS': EnvParameters.N_ACTIONS,
            'EPISODE_LEN': EnvParameters.EPISODE_LEN, 'FOV_SIZE': EnvParameters.FOV_SIZE,
            'WORLD_SIZE': EnvParameters.WORLD_SIZE_LIST,
            'OBSTACLE_PROB': EnvParameters.OBSTACLE_PROB_LIST,
            'ACTION_COST': EnvParameters.ACTION_COST,
            'IDLE_COST': EnvParameters.IDLE_COST, 'GOAL_REWARD': EnvParameters.GOAL_REWARD,
            'AG_COLLISION_COST': EnvParameters.AG_COLLISION_COST,
            'DY_COLLISION_COST': EnvParameters.DY_COLLISION_COST,'NUM_TIME_SLICE':EnvParameters.NUM_TIME_SLICE,
            'lr': TrainingParameters.lr, 'GAMMA': TrainingParameters.GAMMA, 'LAM': TrainingParameters.LAM,
            'CLIPRANGE': TrainingParameters.CLIP_RANGE, 'MAX_GRAD_NORM': TrainingParameters.MAX_GRAD_NORM,
            'ENTROPY_COEF': TrainingParameters.ENTROPY_COEF,
            'VALUE_COEF': TrainingParameters.VALUE_COEF,
            'POLICY_COEF': TrainingParameters.POLICY_COEF,
            'VALID_COEF': TrainingParameters.VALID_COEF,
            'N_EPOCHS': TrainingParameters.N_EPOCHS, 'N_ENVS': TrainingParameters.N_ENVS,
            'N_MAX_STEPS': TrainingParameters.N_MAX_STEPS,
            'N_STEPS': TrainingParameters.N_STEPS, 'MINIBATCH_SIZE': TrainingParameters.MINIBATCH_SIZE,
            'NET_SIZE': NetParameters.NET_SIZE, 'NUM_CHANNEL': NetParameters.NUM_CHANNEL,
            'GOAL_REPR_SIZE': NetParameters.GOAL_REPR_SIZE, 'VECTOR_LEN': NetParameters.VECTOR_LEN,
            'SEED': SetupParameters.SEED, 'USE_GPU_LOCAL': SetupParameters.USE_GPU_LOCAL,
            'USE_GPU_GLOBAL': SetupParameters.USE_GPU_GLOBAL,
            'NUM_GPU': SetupParameters.NUM_GPU, 'RETRAIN': RecordingParameters.RETRAIN,
            'WANDB': RecordingParameters.WANDB,
            'ENTITY': RecordingParameters.ENTITY,
            'TIME': RecordingParameters.TIME, 'EXPERIMENT_PROJECT': RecordingParameters.EXPERIMENT_PROJECT,
            'EXPERIMENT_NAME': RecordingParameters.EXPERIMENT_NAME,
            'EXPERIMENT_NOTE': RecordingParameters.EXPERIMENT_NOTE,
            'SAVE_INTERVAL': RecordingParameters.SAVE_INTERVAL,
            'MODEL_PATH': RecordingParameters.MODEL_PATH}

