

### --- AUTHORS NOTE --- ###
# ...



### --- INITIALIZATION --- ###
from datetime import datetime, timedelta
import pandas as pd
import statistics 
import matplotlib.pyplot as plt
import numpy as np
import math



### --- DATA --- ###
FluxFolderName = "Flux measurements/ES-Cnd [EMAIL]/"
FluxFileName = 'Es-Cnd-Base_Conde_2014_2020_2.csv'
# Which value in the file indicates missing data?
IndMissingData = -9999
# Create a value which will be an indicator of no data occuring
NoDataInd = ''

OutputFolderName = 'ESCND_matrices/'  # End with '/' if you want to specify a folder and make sure the folder actually exists
FileIdentifier = 'ES-Cnd'

Date_Column = 'date'
DateFormat = '%m/%d/%Y' 	# Example: '%d/%m/%Y' for day/month/year
Time_Column = 'time'
Variables = {'LE':'LE_1_1_1', 'G':'G_2_1_1', 'H':'H_1_1_1','Rn':'NETRAD_1_1_1', 'TS':'TS_2_1_1', 'TA':'TA_1_1_1', 'SW_IN':'SW_IN_1_1_1', 'SW_OUT':'SW_OUT_1_1_1'}

CONVERT_TO_MATRIX = False
CONVERT_FROM_MATRIX = True



### --- PROCESSING --- ###
InSituData = pd.read_csv(FluxFolderName+FluxFileName)

#InSituData = InSituData.replace(MissingdataIndicators)
InSituData = InSituData.replace(IndMissingData, NoDataInd)

InSituData[Date_Column] = pd.to_datetime(InSituData[Date_Column], format=DateFormat)
InSituData[Time_Column] = pd.to_timedelta(InSituData[Time_Column]+':00')
# Make sure the data frame is ordered by DATE
InSituData.sort_values(by=Date_Column, ignore_index=True)

#temp = InSituData[[Date_Column, Time_Column]]
#Duplicates = temp.index[temp.duplicated(keep=False)== True].tolist()
#print(temp.loc[Duplicates])



### --- PROCESSING --- ###
if CONVERT_TO_MATRIX:
	for variable_name, column_name in Variables.items():
		Temporary_data = InSituData[[Date_Column, Time_Column, variable_name]]
		Temporary_data = Temporary_data.rename(columns = {variable_name:column_name})
		Temporary_data = Temporary_data.pivot(index=Date_Column, columns=Time_Column, values=column_name)
		# Write new file
		Temporary_data.to_csv(OutputFolderName+FileIdentifier+'_matrix_'+column_name+'.csv')



if CONVERT_FROM_MATRIX:
	first = True
	for variable_name, column_name in Variables.items():
		try:
			Folder = 'ESCND_matrices'
			data = pd.read_csv(Folder+'/'+'ES-Cnd_matrix_'+variable_name+'_filled.csv')
			data.rename(columns={'Row':'date'}, inplace = True)

			data = data.melt(id_vars=["date"], var_name="time", value_name=column_name)
			data = data.sort_values(by=["date", "time"], ignore_index=True)
			data['date'] = pd.to_datetime(data['date'])
			data['date'] = data['date'].dt.strftime('%m/%d/%Y')
			data = data.fillna(int(IndMissingData))

			if first:
				data_frame = data
				first = False
			else:
				data_frame[column_name] = data[column_name]

		except:
			print('Variable ' + variable_name + ' was not present.')
	data_frame['time'] = data_frame.time.str[:-3]
	print(data_frame.head(10))
	data_frame.to_csv(FluxFolderName+FluxFileName[:-4]+'_filled'+'.csv', index=False)

