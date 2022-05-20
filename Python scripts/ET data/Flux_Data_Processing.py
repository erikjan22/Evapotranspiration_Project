

### ----- AUTHORS NOTE ----- ###
# Script for processing and analyzing semi-continuous data from flux stations
### Python file which takes the randomly selected data points on evapotranspiration (ET)  and visualizes and analyzes them ###

from datetime import datetime, timedelta
import pandas as pd
import statistics 
import matplotlib.pyplot as plt
import numpy as np
import math
import Various_Functions




### ----- INITIALIZATION ----- ###

Process_Flux_Data_Daily = False
Process_Flux_Data_Period = False
Evaluate_Data_And_Present_Figures = True

# Define windows size
window_size = 0

# Indicate at which minimum in-situ flux measurements are removed
MinInSituValueThreshold = -25


### ----- DATA ----- ###

# Input: describe source of flux data
FluxFolderName = "Flux measurements/ES-Cnd [EMAIL]/"
FluxFileName = 'Es-Cnd-Base_Conde_2014_2020_2.csv' # original file: 'Es-Cnd-Base_Conde_2014_2020_2.csv'   'Es-Cnd-Base_Conde_2014_2020_2_filled.csv'

# File 3: includes the ET estimations from the Surface Balance Energy (SEB) methods
SEBFolderName = 'Study_sites_results/'
SEBFileName = 'S-SEBI_data_evaluation_ES-Cnd.csv'  # s-sebi file: 'S-SEBI_data_evaluation_ES-Cnd.csv'   eeflux file: 'EEFLUX_data_evaluation_ES-Cnd.csv'

# Which value in the file indicates missing data?
IndMissingData = -9999
# Create a value which will be an indicator of no data occuring
NoDataInd = 'None'

FluxDateColumnName = 'date'
FluxDateFormat = '%m/%d/%Y' 	# Example: '%d/%m/%Y' for day/month/year

# Output: describe output files and folder
OutputFolderName = 'ESCND/'  # End with '/' if you want to specify a folder and make sure the folder actually exists
FileIdentifier = 'ES-Cnd'

# Specify the variables which will be used for evaluation. Do this in the format: <name of variable>: [<variable column in in-sity data>, 'variable column in SEB data']
# The possible names of variables are:
# 	----------------------------------------------------------------------------------------------------------------------------------------------------------------
#	variable name	in-situ column name	S-SEBI column name	EEFlux column name	original in-situ name	description
# 	----------------------------------------------------------------------------------------------------------------------------------------------------------------
#	'Albedo'	'alb_inst'				'Albedo'								'ALB_1_1_1'				Albedo
#	'LST'		'LST_inst'				'LST'				'METRIC_LST'		'TS_2_1_1'				Land surface temperature (K)
#	'LE'		'LE_inst'				'LEi'									'LE_1_1_1'				Instantaneous latent heat flux
#	'G'			'G_inst'				'Gi'									'G_2_1_1'				Instantaneous soil heat flux
#	'H'			'H_inst'				'Hi'									'H_1_1_1'				Instantaneous sensible heat flux
#	'Rn'		'Rn_inst'				'Rni'									'NETRAD_1_1_1'			Net surface radiation flux
#	'ET'		'ET_daily'				'ETd'				'METRIC_ETa'								Daily actual Evapotranspiration (mm/day)

Evaluation_variables = {'Albedo': ['alb_inst', 'Albedo'], \
						'LST': ['LST_inst', 'LST'], \
						'LE': ['LE_inst', 'LEi'], \
						'G': ['G_inst', 'Gi'], \
						'H': ['H_inst', 'Hi'], \
						'Rn': ['Rn_inst', 'Rni'], \
						'ET': ['ET_daily', 'ETd']}

#Evaluation_variables = {'ET': ['ET_daily', 'METRIC_ETa']}


### ----- IMPORTING DATA ----- ###

SEB_data = pd.read_csv(SEBFolderName+SEBFileName)
Landsat_Capture_Dates = pd.DataFrame(SEB_data.DATE.unique(), columns=['DATE'])

# Specifying the columns which are used to compute instantaneous data
Flux_Column_Names_Inst = {'alb':'ALB_1_1_1', 'LST':'TS_2_1_1', 'LE':'LE_1_1_1', 'G':'G_2_1_1', 'H':'H_1_1_1', 'Rn':'NETRAD_1_1_1'}
# Specifying the columns which are used to compute daily data. One of the columns has to be latent heat (LE), since ET is computed from this.
Flux_Column_Names_Daily = {'LE':'LE_1_1_1'}



### ----- PROCESSING FLUX DATA ----- ###

if Process_Flux_Data_Daily:

	# --- PRE-PROCESSING --- #
	# Loading Flux data
	Flux_Data = pd.read_csv(FluxFolderName+FluxFileName)

	# Rename the date column in the flux data
	Flux_Data.rename(columns = {FluxDateColumnName:'DATE'}, inplace = True)
	# Re-format the date column
	Flux_Data['DATE'] = pd.to_datetime(Flux_Data['DATE'], format=FluxDateFormat)

	# Create a new data frame which will function as the repository for the daily results. Intially this dataframe only has the unique date values
	DailyResults = pd.DataFrame(Flux_Data.DATE.unique(), columns=['DATE'])

	# --- INSTANTANEOUS DATA --- #
	# Only select the instantaneous values at specified times, and then take an average from those times.
	TIMES = ["11:30", "12:00"]
	# If you want to add more times, do it here.
	Flux_Data_Time = Flux_Data[(Flux_Data.time==TIMES[0]) | (Flux_Data.time==TIMES[1])]
	Flux_Data_Time = Flux_Data_Time.sort_values(by='DATE', ignore_index=True)
	Flux_Data_Time = Flux_Data_Time.reset_index()

	# Go through all the flux data measurements
	print('\nCOMPUTING INSTANTANEOUS DATA')
	for  var_name, Flux_Column_Name in Flux_Column_Names_Inst.items():
		print('Computing instantanious values for variable "' + var_name + '" stored in column ' + Flux_Column_Name)
		InstantaneousResults = Various_Functions.GET_INSTANTANEOUS_DATA(Flux_Data_Time, Flux_Column_Name, var_name+'_inst', MinInSituValueThreshold, IndMissingData)
		# Replace the count of valid items with a percentage indicator
		column_valid = var_name+'_inst_count_valid'
		InstantaneousResults[var_name+'_inst_percentage_valid'] = (InstantaneousResults[column_valid] / len(TIMES)) * 100 
		InstantaneousResults = InstantaneousResults.drop([column_valid], axis=1)
		# Add the results of the current data to the complete dataframe
		DailyResults = pd.merge(DailyResults, InstantaneousResults, on='DATE')

	# --- DAILY DATA --- #
	print('\nCOMPUTING DAILY DATA')
	# Perform processing for each of the columns of interest
	for var_name, Flux_Column_Name in Flux_Column_Names_Daily.items():
		print('Computing daily values for variable "' + var_name + '" stored in column ' + Flux_Column_Name)
		DailyData = Various_Functions.GET_DAILY_DATA(Flux_Data, Flux_Column_Name, var_name+'_daily', MinInSituValueThreshold, IndMissingData, NoDataInd)
		# Replace the count of valid items with a percentage indicator
		column_valid = var_name+'_daily_count_valid'
		DailyData[var_name+'_daily_percentage_valid'] = (DailyData[column_valid] / 48) * 100 
		DailyData = DailyData.drop([column_valid], axis=1)
		# Add results for the current column of interest to the main results dataframe
		DailyResults = pd.merge(DailyResults, DailyData, on='DATE')

	# Sort the results by date and return 
	DailyResults.sort_values(by='DATE', ignore_index=True)

	# Compute ET from LE
	# From the latent heat column compute the daily total latent heat (MJ per m2 per day)
	DailyResults['ET_daily'] = DailyResults['LE_daily'].replace(NoDataInd, 0).multiply(60*60*24).astype(float).replace(0, NoDataInd)
	# Compute the daily total evapotranspiration (mm per day)
	DailyResults['ET_daily'] = DailyResults['ET_daily'].replace(NoDataInd, 0).divide(1000000).divide(2.45).astype(float).replace(0, NoDataInd)

	print('\nFinished producing daily values. First 10 rows of result: \n', DailyResults.head(10))
	# Write daily data results to csv file
	DailyResults.to_csv(OutputFolderName+FileIdentifier+'_daily.csv', index=False)
	
	# match daily flux values with the landsat capture dates
	Landsat_Capture_Dates_2 = pd.to_datetime(Landsat_Capture_Dates['DATE'], format='%Y/%m/%d')
	DailyResultsMatched = pd.merge(DailyResults, Landsat_Capture_Dates_2, on='DATE')
	print('\nFinished producing matched daily values. First 10 rows of result: \n', DailyResultsMatched.head(10))
	DailyResultsMatched.to_csv(OutputFolderName+FileIdentifier+'_daily_matched.csv', index=False)
	


if Evaluate_Data_And_Present_Figures:
	# Create a time plot with the in-situ daily ET
	DailyResults = pd.read_csv(OutputFolderName+FileIdentifier+'_daily.csv')
	PlottedData = DailyResults[(DailyResults.DATE > "2015-03-01") & (DailyResults.DATE < "2015-11-01")].sort_values(by='DATE', ignore_index=True)
	PlottedData.ET_daily = pd.to_numeric(PlottedData.ET_daily.replace('None', np.nan), downcast="float")
	PlottedData.DATE = pd.to_datetime(PlottedData.DATE)

	plt.plot(PlottedData.DATE, PlottedData.ET_daily, linestyle='--', marker='o', color='b')
	plt.xticks(PlottedData.DATE )
	plt.locator_params(axis='x', nbins=20)
	plt.title('Measured daily evapotranspiration in 2015', fontsize=22)
	plt.xlabel('Time', fontsize = 17)
	plt.ylabel('Daily evapotranspiration [mm/day]', fontsize=17)
	plt.xticks(fontsize = 12, rotation=45)  # Specify x-labels
	plt.yticks(fontsize = 15, rotation=0)  # Specify y-labels
	plt.gcf().subplots_adjust(bottom=0.16) # To make room for all the x-labels
	plt.show()



if Process_Flux_Data_Period:

	Results = pd.read_csv(OutputFolderName+FileIdentifier+'_daily.csv')
	Results['DATE'] = pd.to_datetime(Results['DATE'], format='%Y/%m/%d')

	# --- COMPUTING PERIOD DATA --- #

	# Create a new data frame which will function as the repository for the daily results. Intially this dataframe only has the unique date values
	PeriodResults = Various_Functions.GetLandsatDates(Landsat_Capture_Dates, window_size)
	print(PeriodResults)

	# Collect 
	list_1 = [item+'_inst' for item in list(Flux_Column_Names_Inst.keys())]
	list_2 = [item+'_daily' for item in list(Flux_Column_Names_Daily.keys())]
	All_Final_Columns = list_1 + list_2

	print('\nCOMPUTING PERIOD DATA')
	# Perform processing for each of the columns of interest
	for Column_Name in All_Final_Columns:
		print('Analyzing period data for column ' + Column_Name)
		PeriodData = Various_Functions.GET_PERIOD_DATA(Landsat_Capture_Dates, Results, Column_Name, window_size, NoDataInd)
		# Add results for the current column of interest to the main results dataframe
		PeriodResults = pd.merge(PeriodResults, PeriodData, on='DATE')

	# Compute ET from LE
	# From the latent heat column compute the daily total latent heat (MJ per m2 per day)
	PeriodResults['ET_daily_window_AVG'] = PeriodResults['LE_daily_window_AVG'].replace(NoDataInd, 0).multiply(60*60*24).astype(float).replace(0, NoDataInd)
	# Compute the daily total evapotranspiration (mm per day)
	PeriodResults['ET_daily_window_AVG'] = PeriodResults['ET_daily_window_AVG'].replace(NoDataInd, 0).divide(1000000).divide(2.45).astype(float).replace(0, NoDataInd)

	print('\nFinished producing period values. First 10 rows of result:\n', PeriodResults.head(10))

	# Write period data results to csv file
	PeriodOutputFile = FileIdentifier+'_period_'+str(window_size)+'.csv'
	PeriodResults.to_csv(OutputFolderName+PeriodOutputFile, index=False)




### ----- EVALUATING FLUX DATA ----- ###

if Evaluate_Data_And_Present_Figures:

	# Collect the in-situ data which has been processed in the previous section 
	IN_SITU_data = pd.read_csv(OutputFolderName+FileIdentifier+'_daily_matched.csv')

	# Create lists of the new column names
	IN_SITU_variables = []
	SEB_variables = []

	# Go through the variables which are to be evaluated one-by-one in order to perform some pre-processing
	for var_name, var_column_names in Evaluation_variables.items():
		# Collect the column names
		in_situ_column_name = var_column_names[0]
		seb_column_name = var_column_names[1]
		# Rename the column name of the current variable in the in-situ data
		IN_SITU_data = IN_SITU_data.rename(columns={in_situ_column_name:'IN_SITU_'+var_name})

		# Replace illegitiame values in the in-situ column with an indicator of no data occuring
		if var_name == 'ET':
			IN_SITU_data['IN_SITU_ET'] = IN_SITU_data['IN_SITU_ET'].replace('None', np.nan).astype(float)
		elif var_name == 'LST':
			IN_SITU_data['IN_SITU_LST'] = IN_SITU_data['IN_SITU_LST'].replace(0, np.nan).astype(float) + 273.15
		else:
			IN_SITU_data['IN_SITU_'+var_name] = IN_SITU_data['IN_SITU_'+var_name].replace(0, np.nan).astype(float)

		# Create a new column in the seb data, filled with NaN values
		SEB_data['SEB_'+var_name] = np.nan

		# Go through every row in the SEB dataframe and extract the values of interest
		for index, row in SEB_data.iterrows():
			DictData = eval(row['SEB_mean'].replace("{", "{'").replace("=", "':").replace(", ", ", '").replace("null", "'null'"))
			SEB_data.loc[index, 'SEB_'+var_name] = DictData.get(seb_column_name)

		# Put 'null' values to the right type
		SEB_data['SEB_'+var_name] = SEB_data['SEB_'+var_name].replace('null', np.nan).astype(float)

		# Add elements to the lists of new column names
		IN_SITU_variables += ['IN_SITU_'+var_name]
		SEB_variables += ['SEB_'+var_name]		


	# Merge the Landsat data with the in-situ data. Remove all in-situ data which does not correspond with a date in Landsat_Capture_Dates
	Complete_Data = pd.merge(Landsat_Capture_Dates, IN_SITU_data[['DATE', 'LE_daily_percentage_valid']+IN_SITU_variables], on='DATE', how='inner')

	# Only select the relevant columns and find average for rows with matching dates
	SEB_data = SEB_data[['DATE']+SEB_variables].groupby(SEB_data.DATE).mean()

	Complete_Data = pd.merge(Complete_Data, SEB_data, on='DATE', how='inner')

	print('\nHead of data:', Complete_Data.head(10))
	print('\nTail of data:', Complete_Data.tail(10))

	# Write daily data results to csv file
	Complete_Data.to_csv(OutputFolderName+FileIdentifier+'_complete_ET.csv', index=False)

	### --- PROCESSING --- ###
	EvaluationResults = Various_Functions.EVALUATION(Complete_Data, IN_SITU_variables, SEB_variables, True)

	# Write daily data results to csv file
	EvaluationResults.to_csv(OutputFolderName+FileIdentifier+'_evaluation.csv', index=False)



