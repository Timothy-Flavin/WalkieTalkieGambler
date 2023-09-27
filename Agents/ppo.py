import torch
import torch.nn as nn

# Actor update
# Del_Theta J(Theta) = sum_(t=0) [ Del_Theta * log(Policy_theta(a_t | s_t)) * R(t) ]

# Critic update
# Del_Q = alpha * Del_Theta * (log(policy_Theta(s,a)))*q_w(s,a)
#                                                        ^ Action Value Est


class cnn_A2C(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(torch_rnn, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined) #torch.relu
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
