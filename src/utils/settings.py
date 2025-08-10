
import os
import numpy as np
import random


class Settings:
    @staticmethod
    def set_seed(seed: int = 13) -> None:
        """ ランダムのシードを設定する
            Args:
                seed (int): シード値
            Returns:
                None
        """
        # optional
        # for numpy.random
        np.random.seed(seed)
        # for built-in random
        random.seed(seed)
        # for hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)

