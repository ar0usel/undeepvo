from torch import nn

from .depth_model import DepthNet
from .pose_model import PoseNet
from .utils import init_weights


class UnDeepVO(nn.Module):
    def __init__(self, max_depth=100, min_depth=0.5):
        super(UnDeepVO, self).__init__()
        self.depth_net = DepthNet(max_depth=max_depth, min_depth=min_depth)
        self.pose_net = PoseNet()
        self.apply(init_weights)

    def depth(self, x):
        out = self.depth_net(x)
        return out

    def pose(self, x):
        (out_rotation, out_translation) = self.pose_net(x)
        return out_rotation, out_translation

    def forward(self, x):
        return self.depth(x), self.pose(x)
