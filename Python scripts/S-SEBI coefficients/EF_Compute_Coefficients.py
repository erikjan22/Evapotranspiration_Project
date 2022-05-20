
### --- AUTHORS NOTE --- ###


#### INITIALIZATION ###
import os
import pandas as pd
import numpy as np
import EF_Methods
from datetime import datetime, timedelta

DataFolder = 'Trial_data'   # Make sure there are only .csv S-SEBI results in here
DataFile = ''						  # If no data file is specified, all datafiles in the folder will be processed

AlbedoColumnName = 'Albedo'
LSTColumnName = 'Surface_Temperature'

if DataFile:
	Files = [DataFile]
	print('\nAnalyze data file "' + DataFile + '" in folder "' + DataFolder + '".\n')

else:
	Files = os.listdir(DataFolder+'/')
	print('\nAnalyzing all data files in folder "' + DataFolder + '".\n')


# GENERAL PARAMETERS

### AlbedoQuantile ###
# Specify the fraction of data points which will cut off from the edges of the albedo spectrum
AlbedoQuantile = 0.0005

### LowestAlbedoFraction ###
# Remove lowest end of albedo points
LowestAlbedoFraction = 0.5

### HeighestAlbedoFraction ###
# Remove heighest end of albedo points
HeighestAlbedoFraction = 0.9

### N ###
# State the number of subintervals per interval
N = 10

### MaxStdDev ###
# How much is a maximum element from a subinterval allowed to be below the average value
# The lower the StdDev, the stricter the removal of N points
MaxStdDev = 0.3

### Nmin ###
# How many max elements should there at least be per subinterval
Nmin = 4

### M ###
# How many intervals should there be over the entire range
M = 40

### RSMEmultiplier ###
# How much is a minimum item from an interval allow to be above the prediction line of linear regression
# The lower the RSMEmultiplier, the stricter the removal of M points
RSMEmultiplier = 2

### Mmin ###
# How many min items should there at least be to create the linear regression
# Remember: A lot of the items are already lost due to the cut-off in the edges
Mmin = 10

# Thresholds
LSTminthreshold = 263.15
LSTmaxthreshold = 373.15

# Boolean variable, visualize or not
VISUALIZE_SCATTER_PLOTS = True
WRITE_FILES = False,

# number of samples: sample size
samples = {3:33000, 2:50000, 1:100000} # number of samples and sample size
minNrDataPoints = 50000
Non_Processed_Scenes = {}

EF_Coefficients = pd.DataFrame(columns = ['SCENE_ID', 'DATE', 'File', 'Sample_size', 'Sample', 'DryEdge_a', 'DryEdge_b', 'WetEdge_a', 'WetEdge_b'])

for File in Files:
	DATA = pd.read_csv(DataFolder + '/' + File)
	DATA = DATA[['LANDSAT_SCENE_ID', AlbedoColumnName, LSTColumnName]]
	DATA.rename(columns = {AlbedoColumnName:'Albedo', LSTColumnName:'LST'}, inplace = True) 
	Landsat_ID_DFs = [pd.DataFrame(y) for x, y in DATA.groupby('LANDSAT_SCENE_ID', as_index=False)]
	for SCENE_DATA in Landsat_ID_DFs:
		SCENE_ID = SCENE_DATA.LANDSAT_SCENE_ID.iloc[0]
		Year = int(SCENE_ID[9:13])
		DOY = int(SCENE_ID[13:16])
		Date = datetime(Year, 1, 1) + timedelta(DOY - 1)

		# Filter out all data points which fall above or below the LST thresholds
		NrDataPointsOriginal = len(SCENE_DATA.index)
		SCENE_DATA = SCENE_DATA[(SCENE_DATA.LST > LSTminthreshold) & (SCENE_DATA.LST < LSTmaxthreshold)]
		NrDataPoints = len(SCENE_DATA.index)
		print('\nAnalyzing Landsat scene "%s" of date: %s. Total of %s (out of %s) data points, taken from file "%s".'%(SCENE_ID, Date, NrDataPoints, NrDataPointsOriginal, DataFolder + '/' + File))
		if NrDataPoints < minNrDataPoints:
			# If the current  minimum number of data points
			print('   WARNING: CURRENT LANDSAT SCENE DOES NOT HAVE THE REQUIRED NUMBER OF DATA POINTS, THEREFORE IT IS NOT PROCESSED.')
			Non_Processed_Scenes[File] = SCENE_ID
			continue
		for number_samples, sample_size in samples.items():
			sample_number = 1
			for DATAsample in np.array_split(SCENE_DATA, number_samples):
				if len(DATAsample.index) > sample_size:
					DATAsample = DATAsample.sample(n=sample_size, replace = False)
				else:
					print('   WARNING: NOT ENOUGH DATA POINTS. REPLACE OPTION IS ENABLED.')
					DATAsample = DATAsample.sample(n=sample_size, replace = True)
				DATAsample = DATAsample.reset_index()
				print('      Processing sample: %s, which has %s data points.' %(sample_number, len(DATAsample.index)))
				# Only visualize the first sample of every file if the user wants that to happen
				if VISUALIZE_SCATTER_PLOTS and sample_number == 1:
					VISUALIZE = True
				else:
					VISUALIZE = False
				# Get coefficients for the current file
				[DryEdge_a, DryEdge_b, DryEdge_score, WetEdge_a, WetEdge_b, WetEdge_score] = EF_Methods.TANG_Method(DATAsample, N, MaxStdDev, Nmin, M, RSMEmultiplier, Mmin, 
					AlbedoQuantile, LowestAlbedoFraction, HeighestAlbedoFraction, VISUALIZE)
				# Add results of current file to the main data frame
				EF_Coefficients = EF_Coefficients.append({'SCENE_ID':SCENE_ID, 'DATE':Date, 'File': DataFile, 'Sample_size': sample_size, 'Sample':sample_number, 'DryEdge_a': DryEdge_a, 'DryEdge_b':DryEdge_b, 
					'WetEdge_a':WetEdge_a, 'WetEdge_b':WetEdge_b}, ignore_index=True)
				sample_number += 1

if len(Non_Processed_Scenes) > 0:
	print('\nFinished processing point data of Landsat scenes. The following scenes could not be analyzed due to not having the required number of data points:')
	for file, scene in Non_Processed_Scenes.items():
		print('   Scene "%s" from file "%s"' %(scene, file))

EF_Coefficients = EF_Coefficients.sort_values(by=['DATE'])

EF_Coefficients_Mean = pd.DataFrame(columns = ['DATE', 'SCENE_ID', 'DryEdge_a', 'DryEdge_b', 'WetEdge_a', 'WetEdge_b'])
# For every unique date in the dataframe, compute the mean value of the coefficients 
UniqueIDs = EF_Coefficients.SCENE_ID.unique()
for Landsat_Scene_ID in UniqueIDs:
	EF_Subset = EF_Coefficients[EF_Coefficients.SCENE_ID == Landsat_Scene_ID]
	Date = EF_Subset.DATE.iloc[0]
	EF_Subset_means = EF_Subset[['DryEdge_a', 'DryEdge_b', 'WetEdge_a', 'WetEdge_b']].mean()
	EF_Coefficients_Mean = EF_Coefficients_Mean.append({'DATE':Date, 'SCENE_ID':Landsat_Scene_ID, 'DryEdge_a': EF_Subset_means.DryEdge_a, 
		'DryEdge_b':EF_Subset_means.DryEdge_b, 'WetEdge_a':EF_Subset_means.WetEdge_a, 'WetEdge_b':EF_Subset_means.WetEdge_b}, ignore_index=True)

# Sort the dataframe on date
EF_Coefficients_Mean = EF_Coefficients_Mean.sort_values(by=['DATE'])


if WRITE_FILES:
	# Export the dataframe to a .csv file
	EF_Coefficients.to_csv(DataFolder+'_EF_coefficients.csv', index=False)
	# Export the dataframe to a .csv file
	EF_Coefficients_Mean.to_csv(DataFolder+'_EF_coefficients_Mean.csv', index=False)