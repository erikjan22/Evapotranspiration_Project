
from datetime import datetime, timedelta
import pandas as pd
import statistics 
import matplotlib.pyplot as plt
import numpy as np
import math


Folder = 'ESCND_matrices'
File = 'ES-Cnd_matrix_LE_filled.csv'
data = pd.read_csv(Folder+'/'+File)
data.rename(columns={'Row':'Date'}, inplace = True)

print(data.head(10))

data = data.melt(id_vars=["Date"], 
        var_name="Time", 
        value_name="Value")

#data = pd.to_datetime(InSituData[Date_Column], format=DateFormat)
#data = pd.to_timedelta(InSituData[Time_Column]+':00')
data = data.sort_values(by=["Date", "Time"], ignore_index=True)

print(data.head(10))


data.to_csv(Folder+'/'+'Filled_'+File)

