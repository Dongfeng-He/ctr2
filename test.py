import pandas as pd
from collections import Counter
import math
import json
import numpy as np
import datetime
import pickle
import os


if __name__ == "__main__":
    invite_df = pd.read_csv("/root/ctr2/invite_df.csv",
                                header=None, sep='\t',
                                nrows=100
                                )
    invite_df.to_csv("invite_df_head.csv", index=False, sep='\t')
