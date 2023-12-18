import os

import pandas as pd
import numpy as np


def generate(
        data_path: str,
        output_path: str,
) -> None:
    dfs = []
    fs = os.listdir(data_path)
    for f in fs:
        assert f.endswith('.csv')
        df = pd.read_csv(os.path.join(data_path, f))

