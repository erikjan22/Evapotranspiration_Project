
from datetime import datetime, timedelta
import pandas as pd
import statistics 
import numpy as np
import math
import matplotlib.pyplot as plt


def GET_INSTANTANEOUS_DATA(FLUX_DATA, FLUX_COLUMN_NAME, FINAL_COLUMN_NAME, MinInSituValueThreshold, IndMissingData):
	# Set the date of the previous row initially as the first available date
	previous_date = FLUX_DATA.iloc[0]['DATE']
	column_valid = FINAL_COLUMN_NAME +'_count_valid'
	ResultingData = pd.DataFrame(columns = ['DATE']+[FINAL_COLUMN_NAME])
	column_value = 0
	count_values = 0
	for index, row in FLUX_DATA.iterrows():
	# Get the date and time for the current row
		current_date = row['DATE']
		current_value = row[FLUX_COLUMN_NAME]
		if current_date.year != previous_date.year:
			print('Finished data of year %s, started processing data for year: %s.' %(previous_date.year, current_date.year))
		if current_date != previous_date:
			ResultingData = ResultingData.append({'DATE': previous_date, FINAL_COLUMN_NAME: column_value, column_valid:count_values}, ignore_index=True)
			# Reset values
			column_value = 0
			count_values = 0
			previous_date = current_date
		if current_value != IndMissingData and current_value > MinInSituValueThreshold:
			column_value += current_value
			count_values += 1
	ResultingData = ResultingData.append({'DATE': previous_date, FINAL_COLUMN_NAME: column_value, column_valid:count_values}, ignore_index=True)
	return ResultingData


def GET_DAILY_DATA(FLUX_DATA, FLUX_COLUMN_NAME, FINAL_COLUMN_NAME, MinInSituValueThreshold, IndMissingData, NoDataInd):
	# Create a data frame to hold the data for the current column of interest
	COLUMN_SUM = FINAL_COLUMN_NAME + '_SUM'
	COLUMN_AVG = FINAL_COLUMN_NAME
	COLUMN_VALID = FINAL_COLUMN_NAME +'_count_valid'
	DailyData = pd.DataFrame(columns = ['DATE', COLUMN_SUM, COLUMN_VALID])
	# Set initial values for variables
	Daily_Sum = 0
	count_values = 0
	# Set the date of the previous row initially as the first available date
	previous_date = FLUX_DATA.iloc[0]['DATE']
	# Go through all the flux data measurements
	for index, row in FLUX_DATA.iterrows():
		# Get the date and LE for the current row
		current_date = row['DATE']
		current_value = row[FLUX_COLUMN_NAME]
		# If we go to a new year (ex: from 2014 to 2015) report this to the user. Not necessary, just nice for user to see progress
		if current_date.year != previous_date.year:
			print('Finished data of year %s, started processing data for year: %s.' %(previous_date.year, current_date.year))
		# If the date of the date is not equal to the previous date then we need to store the ET values so dar and reset
		if current_date != previous_date:
			DailyData = DailyData.append({'DATE': previous_date, COLUMN_SUM: Daily_Sum, COLUMN_VALID:count_values}, ignore_index=True)
			# Reset values
			Daily_Sum = 0
			count_values = 0
			previous_date = current_date
		# Add the LE value of the current item to the summation of LE for the current date
		# If the LE value is below zero add nothing instead to the summation
		if current_value != IndMissingData and current_value >= MinInSituValueThreshold:
			Daily_Sum += current_value			
			count_values += 1
		# After going through all the items in the Flux file add the values of the final day
	DailyData = DailyData.append({'DATE': current_date, COLUMN_SUM: Daily_Sum, COLUMN_VALID:count_values}, ignore_index=True)
	# Compute some more variables and add these as rows to the dataframe
	DailyData[COLUMN_VALID] = DailyData[COLUMN_VALID].astype(int)
	DailyData[COLUMN_SUM] = DailyData[COLUMN_SUM].astype(float)
	DailyData[COLUMN_AVG] = DailyData[COLUMN_SUM].divide(DailyData[COLUMN_VALID]).astype(float)
	# We only need two columns: the average and the number of valid values
	DailyData = DailyData[['DATE', COLUMN_VALID, COLUMN_AVG]]
	# Replace NaN values with some indicator of missing data in the last two columns
	DailyData[COLUMN_AVG] = DailyData[COLUMN_AVG].fillna(NoDataInd).replace(0, NoDataInd)
	return DailyData


def FinishPeriod(LIST_ET_VALUES):
	"""Compute the mean, sum and standard deviation from a list of (possibly empty) list of ET values"""
	try:
		ET_AVG = statistics.mean(LIST_ET_VALUES)
		ET_STD = statistics.pstdev(LIST_ET_VALUES)
	except:
		ET_AVG = 0
		ET_STD = 0
	return ET_AVG, ET_STD


def GetLandsatDates(LANDSAT_DATES, WINDOW_SIZE):
	"""Retrieve the date for a certain landsat image using the index variable
	Then compute the corresponding start and end date of the period by defining the WINDOW_SIZE variable with an integer value."""
	LANDSAT_DATES['period_start_date'] = ''
	LANDSAT_DATES['period_end_date'] = ''
	for index, row in LANDSAT_DATES.iterrows():
		landsat_date = LANDSAT_DATES.iloc[index]['DATE'][0:10]
		landsat_date = datetime.strptime(landsat_date,"%Y-%m-%d")
		period_start_date = landsat_date - timedelta(days=WINDOW_SIZE)
		period_end_date = landsat_date + timedelta(days=WINDOW_SIZE)
		#LANDSAT_DATES['DATE'][index] = landsat_date
		LANDSAT_DATES['period_start_date'][index] = period_start_date
		LANDSAT_DATES['period_end_date'][index] = period_end_date
	return LANDSAT_DATES


def GET_PERIOD_DATA(LANDSAT_DATES, DAILY_DATA, COLUMN_NAME, WINDOW_SIZE, NoDataInd):
	"""Aggregate the evapotranspiration (ET) data for the specified period length (WINDOW_SIZE)"""
	# Setup the necessary variables
	Period_Values = []
	AVG_COLUMN_NAME = COLUMN_NAME+"_window_AVG"
	STD_COLUMN_NAME = COLUMN_NAME+"_window_STD"
	PRCT_COLUMN_NAME = COLUMN_NAME+"_percentage_valid_items"

	PeriodData = pd.DataFrame(columns = ['DATE', PRCT_COLUMN_NAME, AVG_COLUMN_NAME, STD_COLUMN_NAME])
	NrLandsatItems = len(LANDSAT_DATES.index)
	index_landsat_dates = 0
	# Create a counter which keeps track of the number of valid values (not 'None' or something like that) in a certain period
	total_percentage_valid_items = 0
	# Retrieve the first set of landsat-date-variables
	current_landsat_date = LANDSAT_DATES.iloc[index_landsat_dates]
	# create a check to ensure that we have passed arrived at the first landsat date
	CHECK_LANDSAT_DATES_ARRIVED = False
	# Go through every date in the ET data set
	for index, row in DAILY_DATA.iterrows():
		# Retrieve the date and ET value for the current row in the ET data set
		current_date = DAILY_DATA[['DATE']].iloc[index, 0]
		current_value = DAILY_DATA[[COLUMN_NAME]].iloc[index,0]
		current_items = DAILY_DATA[[COLUMN_NAME+'_percentage_valid']].iloc[index,0]

		if not CHECK_LANDSAT_DATES_ARRIVED and current_date > current_landsat_date.period_start_date:
			index_landsat_dates += 1
			current_landsat_date = LANDSAT_DATES.iloc[index_landsat_dates]
			continue
		else:
			CHECK_LANDSAT_DATES_ARRIVED = True

		# If the current data has past beyond the period surrounding the current landsat date, we have to go to the next landsat date
		if current_date > current_landsat_date.period_end_date:
			# Compute the percentage of valid items
			fraction_valid_items = total_percentage_valid_items/(WINDOW_SIZE*2+1)
			# Add the results of the current period before starting with a new one
			Average_Value, Std_Value = FinishPeriod(Period_Values)
			PeriodData = PeriodData.append({'DATE':current_landsat_date.DATE, PRCT_COLUMN_NAME:fraction_valid_items, AVG_COLUMN_NAME:Average_Value, 
				STD_COLUMN_NAME:Std_Value}, ignore_index=True)
			# Start a new period by selecting the next landsat and period dates and resetting the ET storage the and valid values counter
			index_landsat_dates += 1
			# If we have run out of dates with landsat images, then we can stop
			if index_landsat_dates >= NrLandsatItems:
				return PeriodData
			current_landsat_date = LANDSAT_DATES.iloc[index_landsat_dates]
			total_percentage_valid_items = 0
			Period_Values = []

		# If we are within the current period append the current ET value
		if current_date >= current_landsat_date.period_start_date:
			# Add current ET to the collection of the current period, but only if there is actually a value (not 'None' or something like that)
			if current_value != NoDataInd:
				Period_Values.append(float(current_value))
				total_percentage_valid_items += current_items
			
	# After going through all the dates in DAILY_DATA finish up by adding the values collected in {Period_Values} as mean and std. dev. to the dataframe
	Average_Value, Std_Value = FinishPeriod(Period_Values)
	fraction_valid_items = total_percentage_valid_items/(WINDOW_SIZE*2+1)
	PeriodData = PeriodData.append({'DATE':current_landsat_date.DATE, PRCT_COLUMN_NAME:fraction_valid_items, AVG_COLUMN_NAME:Average_Value, 
		STD_COLUMN_NAME:Std_Value}, ignore_index=True)
	return PeriodData


def EVALUATION(DATA, GROUNDTRUTHCOLUMNS, ESTIMATIONCOLUMNS, PRESENTOUTPUT=True):
	"""Def:
	Args:
		DATA [pandas.Dataframe]:
		GROUNDTRUTHCOLUMN [string]: A string indicating the column name for the groundtruth data in the DATA set
		ESTIMATIONCOLUMNS [list]: A list of string values, each indicating a column name for a estimation of ET in the DATA set
		PRESENTOUTPUT [Boolean]: Let user indicate whether evaluation results should be presented directly to the user
	Result [dictionary]: Evaluation results for each of the estimation columns
	Example: computation(..., '...', ['...', '...'])
	"""

	# Create a dictionary in which the evaluation results for all the ET estimations will be stored
	EvaluationResults = pd.DataFrame(columns = ['Variables', 'nr_items', 'Rsquared', 'MBE', 'scatter', 'MAE', 'RMSE', 'RMSD', 'NSCE'])

	# For each of the estimations, do the evaluation
	for i in range(len(GROUNDTRUTHCOLUMNS)):
		ESTIMATION = ESTIMATIONCOLUMNS[i]
		GROUNDTRUTH = GROUNDTRUTHCOLUMNS[i]
		print('\nInvestigating %s vs %s' %(GROUNDTRUTH, ESTIMATION))
		# Filter out the dates which have no valid value for either the groundtruth (in-situ) or estimation (seb) data
		data = DATA[['DATE', GROUNDTRUTH, ESTIMATION]].dropna()

		# Select dates and convert them to date time
		dates = data['DATE'].to_list()
		dates = [datetime.strptime(str(i), "%Y-%m-%d") for i in dates]

		# Count the number of items (dates) in the data set
		nr_items = data.shape[0]
		# Check to see that there are valid items
		if nr_items == 0:
			print('NO VALID VALUES IN DATA! Data overview:\n', data)
			continue

		data.to_csv('TEST.csv', index=False)

		O = data[GROUNDTRUTH].to_list()  # O stands for observed value (i.e. ground truth produced from flux data)
		O_mean = np.mean(O)

		M = data[ESTIMATION].to_list()  # m stands for measurement (i.e. ET estimation from landsat data)
		M_mean = np.mean(M)


		## R squared
		sum_nominator = 0
		sum_denominator_1 = 0
		sum_denominator_2 = 0
		
		for index in range(0, nr_items):
			obs_error = O[index] - O_mean
			meas_error = M[index] - M_mean
			sum_nominator += obs_error * meas_error
			sum_denominator_1 += obs_error**2
			sum_denominator_2 += meas_error**2
		
		R_squared = (sum_nominator**2)/(sum_denominator_1*sum_denominator_2)

		## Mean Bias Error (MBE) (a.k.a. Bias), Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
		error_sum = 0
		absolute_error_sum = 0
		squared_error_sum = 0
		for index in range(0, nr_items):
			error = M[index] - O[index]
			error_sum += error
			absolute_error_sum += abs(error)
			squared_error_sum += error**2
		MBE = error_sum / nr_items
		MAE = absolute_error_sum / nr_items
		RMSE = math.sqrt(squared_error_sum / nr_items)

		## Nash-Sutcliffe Coeficcient of Efficiency (NSCE)
		NSCE = (sum_denominator_1 - squared_error_sum) / sum_denominator_1

		# scatter and Root Mean Square Difference (RMSD)
		scatter_sum = 0
		for index in range(0, nr_items):
			scatter_sum += (M[index] - O[index] - MBE) **2
		scatter = scatter_sum / (nr_items - 1)
		
		RMSD = math.sqrt(MBE**2 + scatter**2)

		## Results
		# Store results
		EvaluationResults = EvaluationResults.append({'Variables':ESTIMATION, 'nr_items':nr_items, 'Rsquared':R_squared, 'MBE':MBE, 'scatter':scatter, 
													  'MAE':MAE, 'RMSE':RMSE, 'RMSD':RMSD, 'NSCE':NSCE}, ignore_index=True)
		#Display results to user if requested
		if PRESENTOUTPUT:
			print('Database with %s rows. \
				\nEvaluation results: \
				\n\tRsquared: \t%s	\
				\n\tMBE: \t\t%s 	\
				\n\tscatter: \t%s	\
				\n\tMAE: \t\t%s		\
				\n\tRMSE: \t\t%s	\
				\n\tRMSD: \t\t%s	\
				\n\tNSCE: \t\t%s\n' % (nr_items, R_squared, MBE, scatter, MAE, RMSE, RMSD, NSCE))

			## Visualization
			if 'ET' in ESTIMATION:
				Y_LABEL = 'Evapotranspiration [mm/day]'
				TITLE = 'daily evapotranspiration'
			elif 'LST' in ESTIMATION:
				Y_LABEL = 'Temperature [K]'
				TITLE = 'land surface temperature'
			elif 'Albedo' in ESTIMATION:
				Y_LABEL = 'Albedo'
				TITLE = 'albedo'
			else:
				Y_LABEL = 'Flux [W/m2]'
				if 'LE' in ESTIMATION:
					TITLE = 'latent heat flux'
				elif 'G' in ESTIMATION:
					TITLE = 'soil heat flux'
				elif 'H' in ESTIMATION:
					TITLE = 'sensible heat flux'
				elif 'Rn' in ESTIMATION:
					TITLE = 'net radiation flux'

			title_size = 45
			legend_size = 35
			label_size = 40
			ticks_size = 27.5

			plt.plot_date(dates, O, linestyle='--', marker='o', color='b', label='In-situ measurements', fmt='m')
			plt.plot_date(dates, M, marker='o', color='r', label='SEB estimations', fmt='m')
			plt.legend(loc="upper right", prop={"size":legend_size})
			plt.title('Change over time for %s' %(TITLE), fontsize=title_size)
			plt.xlabel('Time', fontsize = label_size)
			plt.ylabel(Y_LABEL, fontsize=label_size)
			plt.xticks(fontsize = ticks_size, rotation=0)  # Specify x-labels
			plt.yticks(fontsize = ticks_size, rotation=0)  # Specify y-labels
			plt.show()

			min_value = min([min(O), min(M)])
			max_value = max([max(O), max(M)])
			plt.scatter(O, M, marker='x', color='black', s=70)
			plt.plot([min_value,max_value], [min_value,max_value], 'k-', lw=2.5, linestyle='dashed', color='grey')
			plt.title('Scatter plot for %s' %(TITLE), fontsize=title_size)
			plt.xlabel('In-situ measurements', fontsize=label_size)
			plt.ylabel('SEB estimations', fontsize=label_size)
			plt.xticks(fontsize = ticks_size, rotation=0)  # Specify x-labels
			plt.yticks(fontsize = ticks_size, rotation=0)  # Specify y-labels
			plt.show()

	return EvaluationResults