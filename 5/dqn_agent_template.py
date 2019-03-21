import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        """A simple DQN agent"""
        nn.Module.__init__(self)
        self.epsilon = epsilon
        self.n_actions = n_actions
        img_c, img_w, img_h = state_shape

        # Define your network body here. Please make sure agent is fully contained here

        self.conv1 = nn.Conv2d(img_c, 16, kernel_size=5, stride=3, padding=1)
        self.relu1 = nn.ReLU(True)
        self.mp1 = nn.MaxPool2d(3, stride=2)
        self.do1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(True)
        self.mp2 = nn.MaxPool2d(2, stride=2)

        self.head = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(True),
            nn.Linear(32, n_actions),
        )
        self.softmax = nn.Softmax(1)


    def forward(self, state_t):
        """
        takes agent's observation (Variable), returns qvalues (Variable)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        Hint: if you're running on GPU, use state_t.cuda() right here.
        """
        qvalues = state_t
        qvalues = self.mp1(self.relu1(self.conv1(qvalues)))
        qvalues = self.do1(qvalues)
        qvalues = self.mp2(self.relu2(self.conv2(qvalues)))
        qvalues = self.head(qvalues.view(qvalues.size(0), -1))
        qvalues = self.softmax(qvalues)

        assert isinstance(qvalues, Variable) and qvalues.requires_grad, "qvalues must be a torch variable with grad"
        assert len(qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not Variables
        """
        states = Variable(torch.FloatTensor(np.asarray(states)))
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)