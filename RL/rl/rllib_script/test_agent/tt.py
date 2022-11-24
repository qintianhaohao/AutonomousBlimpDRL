import os
import pickle
import json
import numpy as np
import ray
import sys

from ray.tune.trainable.util import TrainableUtil


def find_last_checkpoint(exp_name):
    import time
    def get_FileModifyTime(filePath):
        # '''获取文件的修改时间'''
        t = os.path.getmtime(filePath)
        timeStruct = time.localtime(t)
        # return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)
        return t

    exp_path = os.path.expanduser(f'~/ray_results/{exp_name}')
    for f in os.listdir(exp_path):
        train_paths = {}
        _ = os.path.join(exp_path, f)
        if os.path.isdir(_):
            tt = get_FileModifyTime(_)
            print(tt, type(tt))
            train_paths[tt] = _
        print(train_paths)
        train_paths_sorted = sorted(train_paths.items(),key=lambda x:x[0])
        print(train_paths_sorted)

if __name__ == '__main__':
    find_last_checkpoint('ResidualPlanarNavigateEnv_SAC_test')
