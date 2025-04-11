import datetime

""" Hyperparameters"""


class EnvParameters:
    LOCAL_N_AGENTS_LIST = [8,8,8]  # number of agents used in training
    
    # GLOBAL_N_AGENT_LIST is used to compute the total number of agents.
    # The number is computed as: 
    #    global_num_agent = round(random.choice(GLOBAL_N_AGENT_LIST[cl_num_task]) * grid_area)
    # For a 64x64 grid, grid_area = 4096.
    # To get exactly 5 agents, use the ratio 5/4096.
    GLOBAL_N_AGENT_LIST= [(0.001, 0.002, 0.003)] # [(0.4,0.45,0.5),(0.5,0.55,0.6),(0.6,0.65,0.7)]
    
    N_ACTIONS = 5  # Number of discrete actions available.
    EPISODE_LEN = [356, 356, 356] # Episode length (number of timesteps per episode) for each curriculum level.
    FOV_SIZE = 7  # Field-of-view size (observations are extracted from a square of this size).
    WORLD_SIZE_LIST = [64,96,128] # List of possible grid sizes. To use a 64x64 grid, include 64.
    OBSTACLE_PROB_LIST = [(0.05,0.075,0.1),(0.1,0.125,0.15),(0.15,0.175,0.2)] # Obstacle probabilities for each curriculum level.
    
    # Costs and rewards.
    ACTION_COST = [-0.4,-0.5,-0.6]
    IDLE_COST = [-0.4,-0.5,-0.6]
    ADD_COST=-0.2
    MOVE_BACK_COST=-0.4
    GOAL_REWARD = 0.0
    DY_COLLISION_COST = -1.5
    AG_COLLISION_COST = -1.5
    
    NUM_TIME_SLICE = 9
    WINDOWS = 15
    # Off-route factors for each curriculum level.
    OFF_ROUTE_FACTOR = [0.06, 0.05, 0.04]
    DIS_FACTOR = 0.2
    
    # Timesteps to switch curriculum (if used).
    SWITCH_TIMESTEP = [1e7, 2e7]

    # Utilization window parameters.
    UTI_WINDOWS = list(range(-2, 16))
    UTI_WEIGHT = [0.25, 0.75]  # Weights for edge vs. vertex usage.
    
    OVERALL_WEIGHT = -0.3
    DIS_TIME_WEIGHT = [0.9, 0.1]
    K_STEPS = 5


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
    N_ENVS = 1 # 6 # number of processes  # original: 32
    N_MAX_STEPS = 7e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 4  # number of time steps per process per data collection
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
    ENTITY = 'czimbermark'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'LNS2+RL'
    EXPERIMENT_NAME = 'sztaki-train'
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

