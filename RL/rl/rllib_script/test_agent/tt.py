import os
import pickle
import json
import numpy as np
import ray
import sys

from ray.tune.trainable.util import TrainableUtil

if __name__ == '__main__':
    checkpoint_path = os.path.expanduser(
        "~/ray_results/ResidualPlanarNavigateEnv_PPO_test/PPO_ResidualPlanarNavigateEnv_3cde6_00000_0_2022-11-18_20-12-14/checkpoint_000001"
    )
    # rewrite .metadata file to fit ray rllib v2.1
    metadata = TrainableUtil.load_metadata(checkpoint_path)
    print(metadata)

