from torch import sigmoid,reshape,cat,flatten
from torch.nn.functional import relu
from torch.nn import Module,Linear,LayerNorm,Conv2d,ReLU,Softmax,AdaptiveAvgPool2d,LSTMCell
from torch.nn.init import calculate_gain,orthogonal_,constant_,kaiming_normal_
from torch.cuda.amp.autocast_mode import autocast
from convlstm import ConLSTM

from alg_parameters import *


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Net(Module):
    """network with transformer-based communication mechanism"""

    def __init__(self):
        """initialization"""
        super(Net, self).__init__()
        gain = calculate_gain('relu')

        def init_(m):
            return init(m, orthogonal_, lambda x: constant_(x, 0), gain=gain)

        def init2_(m):
            return init(m, orthogonal_, lambda x: constant_(x, 0), NetParameters.GAIN)

        def init3_(m):
            return init(m, orthogonal_, lambda x: constant_(x, 0))

        self.downsample1 = Conv2d(NetParameters.NUM_CHANNEL+NetParameters.PAST_HIDDEN, NetParameters.NET_SIZE// 2, kernel_size=1, stride=1, bias=False)
        self.conv1 = Conv2d(NetParameters.NUM_CHANNEL+NetParameters.PAST_HIDDEN,NetParameters.NET_SIZE // 2,kernel_size=3,stride=1,padding=1,groups=1,bias=False, dilation=1)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(NetParameters.NET_SIZE // 2,NetParameters.NET_SIZE // 2,kernel_size=3,stride=1,padding=1,groups=1,bias=False, dilation=1)

        self.downsample2 = Conv2d(NetParameters.NET_SIZE // 2, NetParameters.NET_SIZE- NetParameters.GOAL_REPR_SIZE, kernel_size=1, stride=2, bias=False)
        self.conv3 = Conv2d(NetParameters.NET_SIZE // 2,NetParameters.NET_SIZE- NetParameters.GOAL_REPR_SIZE,kernel_size=3,stride=2,padding=1,groups=1,bias=False, dilation=1)
        self.relu2 = ReLU(inplace=True)
        self.conv4 = Conv2d(NetParameters.NET_SIZE- NetParameters.GOAL_REPR_SIZE,NetParameters.NET_SIZE- NetParameters.GOAL_REPR_SIZE,kernel_size=3,stride=1,padding=1,groups=1,bias=False, dilation=1)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.past_convlstm = ConLSTM(NetParameters.NUM_CHANNEL, NetParameters.PAST_HIDDEN, (3, 3), 1, True, True, False)

        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        self.fully_connected_1 = init_(Linear(NetParameters.VECTOR_LEN, NetParameters.GOAL_REPR_SIZE))
        self.fully_connected_2 = init_(Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE))
        self.fully_connected_3 = init_(Linear(NetParameters.NET_SIZE, NetParameters.NET_SIZE))
        self.lstm_memory = LSTMCell(input_size=NetParameters.NET_SIZE, hidden_size=NetParameters.NET_SIZE)
        for name, param in self.lstm_memory.named_parameters():
            if 'bias' in name:
                constant_(param, 0)
            elif 'weight' in name:
                orthogonal_(param)

        # output heads
        self.policy_layer = init2_(Linear(NetParameters.NET_SIZE, EnvParameters.N_ACTIONS))
        self.softmax_layer = Softmax(dim=-1)
        self.value_layer = init3_(Linear(NetParameters.NET_SIZE, 1))

        self.feature_norm = LayerNorm(NetParameters.VECTOR_LEN)
        self.layer_norm_1 = LayerNorm(NetParameters.NET_SIZE - NetParameters.GOAL_REPR_SIZE)
        self.layer_norm_2 = LayerNorm(NetParameters.GOAL_REPR_SIZE)
        self.layer_norm_4 = LayerNorm(NetParameters.NET_SIZE)
        self.layer_norm_5 = LayerNorm(NetParameters.NET_SIZE)

    @autocast()
    def forward(self, obs, vector, input_state):
        """run neural network"""
        num_agent = obs.shape[1]
        obs = reshape(obs, (
        -1, NetParameters.TIME_DEPT, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE))
        curr_obs = obs[:, -1, :, :, :]
        vector = reshape(vector, (-1, NetParameters.VECTOR_LEN))

        x_1=self.past_convlstm(obs)[-1]
        x_1 =cat((x_1, curr_obs), 1)

        identity = self.downsample1(x_1)
        x_1 = self.conv1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.conv2(x_1)
        x_1 += identity
        x_1 = self.relu(x_1)

        identity = self.downsample2(x_1)
        x_1 = self.conv3(x_1)
        x_1 = self.relu2(x_1)
        x_1 = self.conv4(x_1)
        x_1 += identity
        x_1 = self.relu2(x_1)

        x_1 = self.avgpool(x_1)
        x_1 = flatten(x_1, 1)
        x_1 = self.layer_norm_1(x_1)

        # vector input
        x_2=self.feature_norm(vector)
        x_2 =relu(self.fully_connected_1(x_2))
        x_2=self.layer_norm_2(x_2)
        # Concatenation
        x_3 = cat((x_1, x_2), -1)
        h1 = relu(self.fully_connected_2(x_3))
        h1 = self.fully_connected_3(h1)
        h2 = relu(h1 + x_3)
        h2 = self.layer_norm_4(h2)

        # LSTM cell
        memories, memory_c = self.lstm_memory(h2, input_state)
        output_state = (memories, memory_c)
        memories = reshape(memories, (-1, num_agent, NetParameters.NET_SIZE))
        memories =self.layer_norm_5(memories)

        policy_layer = self.policy_layer(memories)
        policy = self.softmax_layer(policy_layer)
        policy_sig = sigmoid(policy_layer)
        value = self.value_layer(memories)
        return policy, value,  policy_sig, output_state,policy_layer


