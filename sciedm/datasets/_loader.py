from importlib.resources import files
from pandas import read_csv

_DATASETS = {
    "block_3sp":           "block_3sp.csv.gz",
    "circle":              "circle.csv.gz",
    "circle_noTime":       "circle_noTime.csv.gz",
    "Lorenz5D":            "Lorenz5D.csv.gz",
    "SumFlow":             "S12CD-S333-SumFlow_1980-2005.csv.gz",
    "sardine_anchovy_sst": "sardine_anchovy_sst.csv.gz",
    "TentMapNoise":        "TentMapNoise.csv.gz",
}

def load_dataset(name: str):
    """
    Parameters
    ----------
    name : str
        Name of dataset to load.

    Returns
    -------
    data : pandas DataFrame
   """

    if name not in _DATASETS.keys():
        msg = f"load_dataset(): Unknown dataset {name!r}. " +\
            f"Available datasets: {list(_DATASETS)}"
        raise ValueError(msg)

    filename  = _DATASETS[name]
    data_path = files("sciedm.datasets.data").joinpath(filename)

    with data_path.open("rb") as f:
        df = read_csv(f, compression="gzip")

    return df
