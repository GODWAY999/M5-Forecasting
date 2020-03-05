import pandas as pd
import numpy as np


def load_data(loc: str = 'data/train.csv') -> np.ndarray:
    data = pd.read_csv(filepath_or_buffer=loc)
    return data.values


def data_split(data: np.ndarray) -> [np.ndarray, np.ndarray]:
    return np.array(data[:, :2]), np.array(data[:, [2]])


def gen_submission(result: np.ndarray,
                   file_name: str = "prediction.csv",
                   loc: str = "./") -> None:
    if loc[-1] != '/':
        loc += '/'

    file = pd.DataFrame(data=result)
    file.to_csv(loc + file_name, header=["time", "open_channels"], index=False, sep=',')
