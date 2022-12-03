import os
import pickle
import json
import numpy as np
import ray
import sys

from ray.tune.trainable.util import TrainableUtil


class YY(object):
    def __init__(self):
        print('yy')

    def tt(self):
        print('yy tt')


class BB(YY):
    def __init__(self):
        super(BB, self).__init__()
        print('bb')

    def tt(self):
        print('bb tt')

class SS(BB):
    def __init__(self):
        super(SS, self).__init__()
        print('ss')

    # def tt(self):
    #     print('ss tt')

if __name__ == '__main__':
    s = SS()
    s.tt()