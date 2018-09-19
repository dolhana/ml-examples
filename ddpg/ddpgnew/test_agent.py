import numpy as np
import gym
import tensorflow as tf
from . import agent
from . import util

def test_main():
    agent.main()

def test_norm_action():
    low = np.array([-2])
    high = np.array([2])
    normer = util.BoxSpaceNormalizer(low, high)
    assert normer.norm([0]) == [0]
    assert normer.norm([-2]) == [-1]
    assert normer.norm([1]) == [.5]

def test_denorm_action():
    low = np.array([-2])
    high = np.array([2])
    normer = util.BoxSpaceNormalizer(low, high)
    assert normer.denorm(normer.norm([0])) == [0]
