import pandas as pd
import numpy as np
from river import drift
from scipy import stats


reference_data = np.random.uniform(0.0, 0.4, size=500)  # Reference window data
target_data = np.random.uniform(0.05, 0.4, size=500)       # Target window data

data = np.concatenate((reference_data, target_data))

ddm_drift_1 = drift.binary.DDM()

for idx, item in enumerate(data):

    ddm_drift_1.update(item)
    if ddm_drift_1.drift_detected:
        print(idx)
