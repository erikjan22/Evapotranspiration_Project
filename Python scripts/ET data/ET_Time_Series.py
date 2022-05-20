

### ---------------------------- ###
### ----- GENERAL COMMENTS ----- ###
### ---------------------------- ###

# VERY GOOD TUTORIAL: https://machinelearningmastery.com/time-series-k-case-study-python-monthly-armed-robberies-boston/




### -------------------------- ###
### ----- INITIALIZATION ----- ###
### -------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
import itertools
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error 

visualize_plots = True
apply_grid_search = False

ColumnOfInterest = 'ETd'
ROIcode = 'ES-Cnd'

# Specify how many of the data points at the end of the (monthly) time series you want to use for validation
# These data points will then not be used for training the time series model
nr_validation_points = 12


Coefficients_type = 'SIGNIFICANCE'


### ------------------------------- ###
### ----- DATA PRE-PROCESSING ----- ###
### ------------------------------- ###

if Coefficients_type != 'BEST_SCORE' and Coefficients_type != 'PLOTS' and Coefficients_type != 'SIGNIFICANCE':
	print('User did not correctly specify which type of coefficients should be chosen:')
	exit()

if ROIcode == "ES-Cnd":
	if Coefficients_type == 'BEST_SCORE':
		optimal_sarima_parameters = [[(0, 0, 0),(1, 1, 2, 12)]]  # Model with the best metric scores
	elif Coefficients_type == 'PLOTS':	
		optimal_sarima_parameters = [[(0, 0, 0),(2, 1, 1, 12)]]  # Model with the coefficients according to acf and pacf
	elif Coefficients_type == 'SIGNIFICANCE':	
		optimal_sarima_parameters = [[(0, 0, 0),(1, 1, 0, 12)]]  # Best scoring model which passes all the significance testing
	max_y_axis = 5.5
	min_y_axis = -0.5
	Date_Template = '%Y-%m-%d'
	AreaName = 'Olive grove'
elif ROIcode == "CA":
	if Coefficients_type == 'BEST_SCORE':
		optimal_sarima_parameters = [[(2, 0, 3),(2, 1, 3, 12)]]
	elif Coefficients_type == 'PLOTS':	
		optimal_sarima_parameters = [[(3, 0, 3),(3, 1, 1, 12)]]
	elif Coefficients_type == 'SIGNIFICANCE':	
		optimal_sarima_parameters = [[(0, 0, 3),(1, 1, 2, 12)]]
	max_y_axis = 5.5
	min_y_axis = -0.5
	Date_Template = '%m/%d/%Y'
	AreaName = 'Agricultural fields in river valley area'
elif ROIcode == "CI":
	if Coefficients_type == 'BEST_SCORE':
		optimal_sarima_parameters = [[(3, 1, 3),(1, 1, 3, 12)]]
	elif Coefficients_type == 'PLOTS':	
		optimal_sarima_parameters = [[(3, 0, 3),(3, 1, 1, 12)]]
	elif Coefficients_type == 'SIGNIFICANCE':	
		optimal_sarima_parameters = [[(0, 0, 3),(1, 1, 2, 12)]]
	max_y_axis = 5.75
	min_y_axis = -0.5
	Date_Template = '%m/%d/%Y'
	AreaName = 'Irrigated agricultural fields in river valley area'
elif ROIcode == "RICE":
	if Coefficients_type == 'BEST_SCORE' or Coefficients_type == 'SIGNIFICANCE':
		optimal_sarima_parameters = [[(0, 0, 1),(2, 1, 3, 12)]] 
	elif Coefficients_type == 'PLOTS':	
		optimal_sarima_parameters = [[(1, 0, 1),(3, 1, 1, 12)]]
	max_y_axis = 8.5
	min_y_axis = -0.5
	AreaName = 'Rice fields'
	Date_Template = '%m/%d/%Y'
elif ROIcode == "GB":
	if Coefficients_type == 'BEST_SCORE' or Coefficients_type == 'SIGNIFICANCE':
		optimal_sarima_parameters = [[(1, 0, 0),(0, 1, 1, 12)]]  # (1, 0, 0),(3, 1, 1, 12) is also quite good and all significant
	elif Coefficients_type == 'PLOTS':	
		optimal_sarima_parameters = [[(1, 0, 1),(3, 1, 1, 12)]]
	max_y_axis = 4.75
	min_y_axis = -0.5
	Date_Template = '%Y-%m-%d'
	AreaName = 'Agricultural fields in Guadalquivir basin'
else:
	print('User did not select a correct ROI code.')
	exit()


# Specify the data file
DataFolder = 'Study_sites_results'
DataFile = 'S-SEBI_data_results_' + ROIcode + '.csv'
print('\nAnalyzing time series data of file: "%s"\nThis is area: "%s".\n' %(DataFile, AreaName))

# -------------------------------------------------------------------------
# Location							| ROIcode	| File name
# -------------------------------------------------------------------------
# ES-Cnd  		  					| ES-Cnd  	| 'S-SEBI_data_results_ES-Cnd.csv'				
# Central agricultural				| CA 		| 'S-SEBI_data_results_CA.csv'
# Irrigated central agriculture 	| CI 		| 'S-SEBI_data_results_CI.csv'
# Rice fields	  					| RICE 		| 'S-SEBI_data_results_RICE.csv'
# Complete Guadalquivir basin 		| GB 		| 'S-SEBI_data_results_GB.csv'

# Read the data file and rename some of the columns
def parser(x):
	return datetime.strptime(x, Date_Template)

def actual_value(value):
	new_value = value.partition(ColumnOfInterest+"=")[2]
	new_value = new_value.partition("}")[0]
	return new_value
data_raw = read_csv(DataFolder+'/'+DataFile, parse_dates=['DATE'], date_parser=parser).rename(columns={'system:index':'SCENE_ID'})
data_raw['Landsat'] = data_raw['SCENE_ID'].str[0:1] + data_raw['SCENE_ID'].str[3:4]
# Go through the columns and 
data_raw['ETd'] = data_raw['SEB_mean'].apply(actual_value)
data_raw['ETd_std'] = data_raw['SEB_std'].apply(actual_value)
data_raw['nrPixels'] = data_raw['nrPixels'].apply(actual_value)

# Remove all rows which don't have actual values
# For some (unknown) reason sometimes the ETd_mean has an actual value, while ETd_std doesn't have one
data_filtered = data_raw[(data_raw.ETd!="null") & (data_raw.ETd_std!="null")]
data_filtered = data_filtered[pd.to_numeric(data_filtered.ETd, errors='coerce').isnull() == False]
data_filtered.ETd = pd.to_numeric(data_filtered.ETd, downcast="float")
data_filtered.ETd_std = pd.to_numeric(data_filtered.ETd_std, downcast="float")

# Create a monthly mean
data_monthly_mean = data_filtered.groupby([data_filtered["DATE"].dt.year, data_filtered["DATE"].dt.month])["ETd"].mean() \
	.to_frame().reset_index(level=[1]).rename(columns={'DATE':'Month', 'ETd':'ETmean'}).reset_index(level=[0]).rename(columns={'DATE':'Year'})
data_monthly_std = data_filtered.groupby([data_filtered["DATE"].dt.year, data_filtered["DATE"].dt.month])["ETd"].std() \
	.to_frame().reset_index(level=[1]).rename(columns={'DATE':'Month', 'ETd':'ETstd'}).reset_index(level=[0]).rename(columns={'DATE':'Year'})
data_monthly = data_monthly_mean 
data_monthly['ETstd'] = data_monthly_std['ETstd']


# Go through monthly data row by row and fill in missing values
time_series = pd.DataFrame(columns = ['Year', 'Month', 'ETmean', 'ETstd'])
start_year = 2000
end_year = 2020
for y in range(start_year, end_year+1):
	for m in range(1,13):
		# See if a row can be found in the data (???)
		data_row = data_monthly[(data_monthly.Year == y) & (data_monthly.Month == m)]
		if len(data_row.index) == 1: 
			time_series = time_series.append(data_row, ignore_index=True)
		elif len(data_row.index) == 0: 
			time_series = time_series.append({'Year':y, 'Month':m, 'ETmean': None, 'ETstd': None}, ignore_index=True)
		else:
			print('WARNING! THIS SHOULD NOT OCCUR!')

# Split time_series data into a collection for training and a collection for validation (i.e. testing)
data_devision_point = len(time_series) - nr_validation_points
training_data, validation_data = time_series[0:data_devision_point], time_series[data_devision_point:]
print('Actual size [possible size]: \t\tComplete (monthly) time series: %s [%s] \t\tSize training data: %s [%s] \t\tSize validation data: %s [%s]' \
		% (len(time_series.ETmean.dropna()), len(time_series), len(training_data.ETmean.dropna()), len(training_data), len(validation_data.ETmean.dropna()), len(validation_data)))


# Present data so far if specified by the user
if visualize_plots:
	# In this section the complete time_series data will be used
	print('\nRaw data:\n', data_filtered.head())
	data_filtered_L5 = data_filtered[data_filtered.Landsat == 'L5']
	data_filtered_L7 = data_filtered[data_filtered.Landsat == 'L7']
	data_filtered_L8 = data_filtered[data_filtered.Landsat == 'L8']

	# Create plots with raw data
	fig, (ax1, ax2) = plt.subplots(2)#, sharex=True, sharey=True)

	legend_size = 20
	x_label_size = 20
	y_label_size = 15
	tick_size = 18

	fig.text(0.5, 0.9, 'Measured daily evapotranspiration over time', ha='center', fontsize=24, fontweight='bold')
	fig.text(0.5, 0.05, 'Date', ha='center', fontsize=x_label_size, fontweight='bold')
	fig.text(0.09, 0.5, 'Daily evapotranspiration [mm/day]', va='center', rotation='vertical', fontsize=x_label_size, fontweight='bold')

	ax1.scatter(data_filtered_L5.DATE, data_filtered_L5.ETd, c='black', s = 15)
	ax1.plot(data_filtered_L5.DATE, data_filtered_L5.ETd, color='red',linestyle='dashed',linewidth=2, label="Landsat 5")
	ax1.scatter(data_filtered_L8.DATE, data_filtered_L8.ETd, c='black', s = 15)
	ax1.plot(data_filtered_L8.DATE, data_filtered_L8.ETd, color='blue',linestyle='dashed',linewidth=2, label="Landsat 8")
	ax1.legend(loc="upper right", prop={'size': legend_size})
	ax1.tick_params(axis='both', which='major', labelsize=17)
	#ax1.set_xlabel('Date', fontsize=x_label_size, fontweight='bold')
	#ax1.set_ylabel('Daily evapotranspiration [mm/day]', fontsize=y_label_size, fontweight='bold')
	#ax1.set_title('TITEL-1')

	ax2.scatter(data_filtered_L7.DATE, data_filtered_L7.ETd, c='black', s = 15)
	ax2.plot(data_filtered_L7.DATE, data_filtered_L7.ETd, color='green',linestyle='dashed',linewidth=2, label="Landsat 7")
	ax2.legend(loc="upper right", prop={'size': legend_size})
	ax2.tick_params(axis='both', which='major', labelsize=17)
	#ax2.set_xlabel('Date', fontsize=x_label_size, fontweight='bold')
	#ax2.set_ylabel('Daily evapotranspiration [mm/day]', fontsize=y_label_size, fontweight='bold')
	#ax2.set_title('TITEL-2')
	plt.show()

	print('\nOverview of monthly averaged data:\n', time_series.head())

	plt.errorbar(time_series.index, time_series.ETmean, time_series.ETstd, linestyle='None', c='black', marker='o')
	#plt.scatter(time_series.index, time_series.ETmean, c='black', s = 15)
	plt.plot(time_series.index, time_series.ETmean, color='blue',linestyle='dashed',linewidth=2)
	plt.title('Monthly average of daily evapotranspiration [mm/day]')
	plt.xlabel('Month index')
	plt.ylabel('Daily Evapotranspiration [mm/day]')
	plt.show()

	print('\nDataframe has %s entries of which %s are missing/NaN.' %(len(time_series.index), time_series.ETmean.isnull().sum()))
	print('Further description of data:\n', time_series.ETmean.describe())

	### ACF AND PACF
	# Define the number of lags used in ACF and PACF plots
	NrLags = 40
	# Instructions and tips: 
	# https://stats.stackexchange.com/questions/166454/use-acf-and-pacf-for-irregular-time-series
	# https://stats.stackexchange.com/questions/242119/how-can-i-interpolate-a-time-series-subject-to-stochastic-perturbation

	# Let's plot the ACF and PACF to investigate the time series data.
	# One issue is that the data has a lot of missing values. These can be removed using the .dropna() method.
	# However, this would influence the plots, since the data would be 'shifted' due to values dissapearing.
	# Instead, interpolation is used.
	time_series_filled = time_series.interpolate(method='linear')
	# Find remaining null values
	print('\nRemaining null values after interpolation:', time_series_filled[time_series_filled.ETmean.isnull()])
	# Replace remaining null values with a value (this shouldn't have to be used, since all the missing values should already have been filled during interpolation)
	time_series_filled.ETmean = time_series_filled.ETmean.fillna(0.3)
	print('\nInterpolated DataFrame has %s entries of which %s are missing/NaN.' %(len(time_series_filled.index), time_series_filled.ETmean.isnull().sum()))
	print('Further description of data:\n', time_series_filled.ETmean.describe())

	# Differencing the time series data to remove periodicity and ensure stationarity
	data_seasonal_diff = time_series_filled.ETmean.squeeze().diff(12)
	data_double_diff = data_seasonal_diff.diff(1)
	time_series_filled['seasonal_diff'] = data_seasonal_diff.squeeze()
	time_series_filled['double_diff'] = data_double_diff.squeeze()
	print('Filled time series with differenced values:\n', time_series_filled.head(25))

	# Create a ACF and PACF of the time series data
	fig1, ax1 = plt.subplots(2, 3, figsize=(10,5))
	# Use the squeeze() method to change a DataFrame with one column into a Series object
	ax1[0,0].plot(time_series_filled.ETmean.squeeze(), c = 'r')
	ax1[0,0].set_title('Monthly aggregated - No differencing')
	plot_acf(time_series_filled.ETmean.squeeze(), lags = NrLags, ax=ax1[0,1])
	ax1[0,1].set_title('ACF - No differencing')
	plot_pacf(time_series_filled.ETmean.squeeze(), lags = NrLags, ax=ax1[0,2])
	ax1[0,2].set_title('PACF - No differencing')

	ax1[1,0].plot(data_seasonal_diff, c='g')
	ax1[1,0].set_title('Monthly aggregated - Seasonal differencing [12]')
	plot_acf(data_seasonal_diff.dropna(), lags = NrLags, ax=ax1[1,1])
	ax1[1,1].set_title('ACF - Seasonal differencing [12]')
	plot_pacf(data_seasonal_diff.dropna(), lags = NrLags, ax=ax1[1,2])
	ax1[1,2].set_title('PACF - Seasonal differencing [12]')

	#ax1[2,0].plot(data_double_diff, c='b')
	#ax1[2,0].set_title('Plot - Double diff [Diff(12), Diff(1)]')
	#plot_acf(data_double_diff.dropna(), lags = NrLags, ax=ax1[2,1])
	#ax1[2,1].set_title('ACF - Double diff [Diff(12), Diff(1)]')
	#plot_pacf(data_double_diff.dropna(), lags = NrLags, ax=ax1[2,2])
	#ax1[2,2].set_title('PACF - Double diff [Diff(12), Diff(1)]')

	#fig1.suptitle('ACF and PACF graphs of (un)differenced, interpolated and aggregated (i.e. monthly mean) daily evapotranspiration time series.')
	plt.show()




### ---------------------------------------- ###
### ----- FITTING SEASONAL ARIMA MODEL ----- ###
### ---------------------------------------- ###

# LINK TO INSTRUCTIONS: https://towardsdatascience.com/how-to-forecast-sales-with-python-using-sarima-model-ba600992fa7d

# There are several convergence methods. These methods might effect the outcome, in that AIC, BIC and Log Likelihood scores differ for the same model.
# The different convergence models greatly differ in the amount of time they take.
# Options:
# 	'newton' [Newton-Raphson]: Very slow convergence, do not put the maximum number of iterations too.
# 	'nm' [Nelder-Mead, aka Simplex-method]: Make sure the maximum number of iterations is high, else the method might not converge (e.g. 1000). Fast convergence.
#	'bfgs' [Broyden-Fletcher-Goldfarb-Shanno (BFGS)]: 
# 	'lbfgs' [limited-memory BFGS with optional box constraints]:
# 	'powell' [modified Powell’s method]: Useful method, doesn't take too much time to run with low number of MaxIterations (e.g. 100).
# 	'cg' [conjugate gradient]:
# 	'ncg' [Newton-conjugate gradient]:
# 	'basinhopping' [global basin-hopping solver]:

# Documentation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

# CONVERGENCE SETTINGS
arima_method = 'powell'
max_iterations = 100  # The number of iterations which are allowed before convergence has to be attained
start_index = 12
nr_forecast_points = 24


### -------------------------------- ###
### ----- IN-SAMPLE PREDICTION ----- ###
### -------------------------------- ###

# I am using the .get_prediction() and .get_forecast() methods for in-sample predictions (time frame used to fit the model) and out-of-sample forecasting (after the model time frame) respectively.
# In contrast to the .predict() and .forecast() methods, these methods allow for multi-step prediction/forecasting and return additional results (e.g. confidence interval).
# However, the main results for Y_prediction = results.get_prediction(start=start_index) and Y_prediction = results.predict(start=start_index) are the same.
# I will be using one-step ahead predictions/forecasting.
# Documentation: 	
#	https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.predict.html
# 	https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.get_prediction.html
#	https://machinelearningmastery.com/multi-step-time-series-forecasting/


def time_series_predictions(results, training_data, validation_data, start_index, nr_forecast_points):
	""""Def: function which creates a prediction and a forecast using a particular ARIMA model.
	Args:
		start_index: Exclude the "true" and predicted values for the first two years, since these are insufficient
		nr_forecast_points: Number of months after the complete time_series (the future) which will be forecasted
	"""
	### IN-SAMPLE PREDICTION ###
	Y_in_sample = training_data.ETmean.iloc[start_index:]
	# Create a prediction. The start parameter indicates when the prediction starts (index of the time_series dataframe)
	Y_prediction = results.get_prediction(start=start_index)
	# Remove all NaN values from the "true" dataset
	Y_in_sample_valid = Y_in_sample[Y_in_sample.isna() != True]
	# Only select those data points for the predicted dataset which correspond with the indexes of remaining data points in the "true" dataset 
	Y_prediction_valid = Y_prediction.predicted_mean[Y_in_sample_valid.index]
	# compute the Mean Standard Error
	MSE_in_sample = mean_squared_error(Y_in_sample_valid, Y_prediction_valid) 


	### OUT-OF-SAMPLE FORECAST ###
	Y_out_of_sample = validation_data.ETmean
	# Create a forecast
	Y_forecast = results.get_forecast(steps=nr_validation_points+nr_forecast_points)  # 'steps' parameter is the number of months which will be forcasted
	# Remove all NaN values from the "true" dataset
	Y_out_of_sample_valid = Y_out_of_sample[Y_out_of_sample.isna() != True]
	# Only select those data points for the predicted dataset which correspond with the indexes of remaining data points in the "true" dataset 
	Y_forecast_valid = Y_forecast.predicted_mean[Y_out_of_sample_valid.index]
	# compute the Mean Standard Error
	MSE_out_of_sample = mean_squared_error(Y_out_of_sample_valid, Y_forecast_valid) 

	return MSE_in_sample, MSE_out_of_sample, Y_in_sample_valid, Y_out_of_sample_valid, Y_prediction, Y_forecast

	


if apply_grid_search:
	print('\nAPPLY GRID SEARCH TO FIND OPTIMAL HYPERPARAMETERS FOR SARIMA TIME SERIES MODEL.')
	# COEFFICIENT SETTINGS
	p = q = range(0, 4)
	d = range(0, 2)
	pdq = list(itertools.product(p, d, q))
	P = Q = range(0, 4)
	D = range(0, 2)
	s = 12
	seasonal_PDQ = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]
	# Create overview of results
	models_overview = pd.DataFrame(columns = ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'AIC', 'BIC', 'LogLikelihood', 'MSE_in_sample', 'MSE_out_of_sample'])
	best_score = {'param':None, 'param_seasonal':None, 'AIC':99999}

	for param in pdq:
		for param_seasonal in seasonal_PDQ:
			p = param[0]
			d = param[1]
			q = param[2]
			P = param_seasonal[0]
			D = param_seasonal[1]
			Q = param_seasonal[2]
			s = param_seasonal[3]
			try:
				mod = sm.tsa.statespace.SARIMAX(training_data.ETmean, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
				results = mod.fit(maxiter=max_iterations, method=arima_method, disp=False)
				MSE_in_sample, MSE_out_of_sample = time_series_predictions(results, training_data, validation_data, start_index, nr_forecast_points)
				# For results see: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html
				# "disp" is a Boolean parameter, default value is True. If True, a lot of (additional) info is provided to the user.
				print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
				# Add results to overview
				models_overview = models_overview.append({'p':p, 'd':d, 'q':q, 'P':P, 'D':D, 'Q':Q, 's':s, 'AIC':results.aic, 'BIC':results.bic, 'HQIC':results.hqic, \
					'LogLikelihood':results.llf, 'MSE_in_sample':MSE_in_sample, 'MSE_out_of_sample':MSE_out_of_sample}, ignore_index=True)
				# Check to see if the new results are an improvement 
				if results.aic < best_score.get('AIC'):
					best_score = {'param':param, 'param_seasonal':param_seasonal, 'AIC':results.aic}
			except:
				print('ARIMA{}x{} - NOT EXECUTED PROPERLY'.format(param, param_seasonal))
				continue
	

	print('\nThe model with the lowest AIC score:\n', 'ARIMA{}x{}12 - AIC:{}'.format(best_score.get('param'), best_score.get('param_seasonal'), best_score.get('AIC')))
	best_model = sm.tsa.statespace.SARIMAX(training_data.ETmean, order=best_score.get('param'), seasonal_order=best_score.get('param_seasonal'), enforce_stationarity=False, enforce_invertibility=False)
	# Recreate the best model using the optimal parameters
	results = best_model.fit(maxiter=max_iterations, method=arima_method, disp=False)
	print('Summary of results for best model:\n', results.summary())
	# Save results
	models_overview.to_csv('Time_models_'+DataFile, index=False)

else:
	for parameter_set in optimal_sarima_parameters:
		param = parameter_set[0]
		param_seasonal = parameter_set[1]
		print('\n\n--- ARIMA{}x{} ---'.format(param,param_seasonal))
		best_model = sm.tsa.statespace.SARIMAX(training_data.ETmean, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
		results = best_model.fit(maxiter=max_iterations, method=arima_method, disp=False)
		print('Summary of results for model:\n', results.summary())
		print('t-values:\n', results.tvalues)
		MSE_in_sample, MSE_out_of_sample, Y_in_sample_valid, Y_out_of_sample_valid, Y_prediction, Y_forecast  = time_series_predictions(results, training_data, validation_data, start_index, nr_forecast_points)
		print('\nMean Squared Error of out-of-sample data: \t%s\t\t Number of data points: %s' %(MSE_out_of_sample, len(Y_out_of_sample_valid)))
		print('\nMean Squared Error of in-sample data: \t%s\t\t\t Number of data points: %s' %(MSE_in_sample, len(Y_in_sample_valid)))


# Visualization of diagnostics
if visualize_plots:
	results.plot_diagnostics(figsize=(18, 8))
	plt.show()

## RESIDUALS ##
# line plot of residuals
#residuals = DataFrame(results.resid)
#residuals.plot()
#plt.title('Residuals')
#plt.show()
# density plot of residuals
#residuals.plot(kind='kde')
#plt.title('Density plot of residuals')
#plt.show()
# summary stats of residuals
#print('Summary statistics of residuals', residuals.describe())



### ------------------------------------- ###
### ----- VISUALIZATION OF ALL DATA ----- ###
### ------------------------------------- ###

if visualize_plots:
	tick_size = 19
	label_size = 20
	legend_size = 20

	
	#ax_1 = time_series.ETmean.plot(label='Aggregated monthly ET', linestyle='dashed')

	# PLOT SPECIFICATION
	#ax_1.set_xlabel('Month', fontsize=label_size, fontweight='bold')
	#ax_1.set_ylabel('Daily evapotranspiration [mm/day]', fontsize=label_size, fontweight='bold',)
	#ax_1.set_title('Evapotranspiration time series & forecast', fontsize= 30)
	#plt.ylim([min_y_axis, max_y_axis])
	#plt.xticks(size = tick_size)
	#plt.yticks(size = tick_size)
	#plt.legend(prop={"size":legend_size})
	#plt.show()


	#ax_2 = time_series.ETmean.plot(label='Aggregated monthly ET', linestyle='dashed')

	# VISUALIZATION OF WITHIN STUDY-PERIOD PREDICTION VALUES
	#Y_prediction.predicted_mean.plot(ax=ax_2, label='In-sample prediction', alpha=.8, figsize=(14, 4), color='purple')
	#prediction_ci = Y_prediction.conf_int()  # confidence area?
	#ax_2.fill_between(prediction_ci.index, prediction_ci.iloc[:, 0], prediction_ci.iloc[:, 1], color='k', alpha=.2)

	# PLOT SPECIFICATION
	#ax_2.set_xlabel('Month', fontsize=label_size, fontweight='bold')
	#ax_2.set_ylabel('Daily evapotranspiration [mm/day]', fontsize=label_size, fontweight='bold')
	#ax_2.set_title('Evapotranspiration time series & forecast', fontsize= 30)
	#plt.ylim([min_y_axis, max_y_axis])
	#plt.xticks(size = tick_size)
	#plt.yticks(size = tick_size)
	#plt.legend(prop={"size":legend_size})
	#plt.show()


	ax_3 = time_series.ETmean.plot(label='Aggregated monthly ET', linestyle='dashed')

	# VISUALIZATION OF WITHIN STUDY-PERIOD PREDICTION VALUES
	Y_prediction.predicted_mean.plot(ax=ax_3, label='In-sample prediction', alpha=.8, figsize=(14, 4), color='purple')
	prediction_ci = Y_prediction.conf_int()  # confidence area?
	ax_3.fill_between(prediction_ci.index, prediction_ci.iloc[:, 0], prediction_ci.iloc[:, 1], color='k', label='95% confidence interval.', alpha=.2)

	# VISUALIZATION OF FUTURE PREDICTION
	Y_forecast.predicted_mean.plot(ax=ax_3, label='Out-of-sample forecast')
	forecast_ci = Y_forecast.conf_int()
	ax_3.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='k', alpha=.2)

	# PLOT SPECIFICATION
	ax_3.set_xlabel('Month', fontsize=label_size, fontweight='bold')
	ax_3.set_ylabel('Daily evapotranspiration [mm/day]', fontsize=label_size, fontweight='bold')
	ax_3.set_title('Evapotranspiration time series & forecast', fontsize= 30)
	plt.ylim([min_y_axis, max_y_axis])
	plt.xticks(size = tick_size)
	plt.yticks(size = tick_size)
	plt.legend(prop={"size":legend_size})
	plt.show()
