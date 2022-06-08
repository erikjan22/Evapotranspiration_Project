
"""MAPPING TIME-SERIES EVAPOTRANSPIRATION FOR AGRICULTURAL APPLICATIONS
A CASE STUDY IN GUIDALQUIVIR RIVER BASIN
Full report: http://uu.diva-portal.org/smash/get/diva2:1579426/FULLTEXT01.pdf
Erik Bootsma - June 2021 - Uppsala University
Contact: erik_jan_22@hotmail.com

AUTHORS NOTE:
This script is used to store two functions which use an edge-fitting algorithm to compute EF-coefficients."""


def TANG_Method(DATA, N, MaxStdDev, Nmin, M, RSMEmultiplier, Mmin, AlbedoQuantile, LowestAlbedoFraction, HeighestAlbedoFraction, DisplayResults=False):
	"""Def: Compute the coefficients of the evaporative fraction (EF) using the TANG method.
	Args:
		M [int]: The number of intervals. Tang advices M <= 20.
		N [int]: The number of subintervals in each interval. Tang advices N >= 5.
		Mmin [int]: The minimum allowed number of intervals.
		Nmin [int]: The minimum allowed number of subintervals
		MaxStdDevDry [float]: 
		AlbedoQuantile [float]: Value between 0 and 1 indicating the fraction of extreme albedo values which should be ignored at the start and end of the albedo scale.
	Returns:
		...
	Example:
		..."""

	### INITIALIZATION ###
	import matplotlib.pyplot as plt
	import pandas as pd
	from sklearn.linear_model import LinearRegression
	import numpy as np
	from math import sqrt
	from sklearn.metrics import mean_squared_error

	def Compute_Edge(EDGE_TYPE):
		# Decide which edge to detect
		if EDGE_TYPE == 'DRY_EDGE':
			DRY_EDGE = 1
		elif EDGE_TYPE == 'WET_EDGE':
			DRY_EDGE = 0
		else:
			return 'ERROR in function "Compute_Edge"! Should be called using argument "DRY_EDGE" or "WET_EDGE".'
			
		# Create a range of albedo values
		StartAlbedo = DATA.Albedo.quantile(AlbedoQuantile)
		StopAlbedo = DATA.Albedo.quantile(1-AlbedoQuantile)
		# Compute the albedo step size between the different intervals
		AlbedoIntervalSize = (StopAlbedo - StartAlbedo) / M
		# Compute the albedo step size between different subintervals
		AlbedoSubintervalSize = AlbedoIntervalSize / N
		# Create a range of albedo values which will be inspected
		AlbedoIntervals = np.arange(StartAlbedo, StopAlbedo, AlbedoIntervalSize)

		### PART I ###
		# Find the extreme (maximum or minimum) LST items in each of the intervals
		# Create a dataframe to save the optimal LST and corresponding Albedo values for each of the intervals 
		OptimalItems = pd.DataFrame(columns = ['LSTopt', 'AlbedoOpt'])

		# Go through all the intervals in the albedo range
		for Albedo_interval in AlbedoIntervals:
			# Select the data within the current interval
			DATAinterval = DATA[(DATA.Albedo >= Albedo_interval) & (DATA.Albedo < Albedo_interval+AlbedoIntervalSize)]
			# For the current interval, create a series of subintervals
			AlbedoSubintervals = np.arange(Albedo_interval, Albedo_interval+AlbedoIntervalSize, AlbedoSubintervalSize)
			# Create a collection of extreme data points
			ExtItemsInterval = pd.DataFrame(columns = ['LSText', 'AlbedoExt'])
			# Fo through each subinterval
			for Albedo_subinterval in AlbedoSubintervals:
				# Select the data within the current subinterval
				DATAsubinterval = DATAinterval[(DATAinterval.Albedo >= Albedo_subinterval) & (DATAinterval.Albedo < Albedo_subinterval+AlbedoSubintervalSize)]
				# If there are items in the current subinterval, process these items 
				if len(DATAsubinterval.index) > 0:
					# Find the extreme LST value and the corresponding albedo value for each subinterval
					# If we're looking for the dry edge, we will select the maximum values. If we're looking for the wet edge, find the minimum values
					if DRY_EDGE:
						EXTvalue = DATAsubinterval.loc[DATAsubinterval['LST'].idxmax()]
					else:
						EXTvalue = DATAsubinterval.loc[DATAsubinterval['LST'].idxmin()]
					ExtItemsInterval = pd.concat([ExtItemsInterval, pd.DataFrame({'LSText':EXTvalue.LST, 'AlbedoExt':EXTvalue.Albedo}, index=[0])], ignore_index=True)
			# Count the number of items in the collection of extreme data points
			Ncurrent = len(ExtItemsInterval.index)
			# Compute the average of the extreme LST values
			LSTavg = ExtItemsInterval.LSText.mean()
			# Compute the standard deviation of the extreme LST values
			LSTstdDev = ExtItemsInterval.LSText.std()
			# Create a check
			CHECK = False
			# While we haven't reached the minimum number of subintervals or the standard deviation is still large enough
			while Ncurrent > Nmin and LSTstdDev > MaxStdDev and CHECK == False:
			# Go through all the extreme elements and see if any of them need to be dropped
			# The statement has to be true for elements to be kept!
				if DRY_EDGE:
					ExtItemsIntervalNew = ExtItemsInterval[(ExtItemsInterval.LSText>LSTavg-LSTstdDev) & (ExtItemsInterval.LSText < LSTavg + 2*LSTstdDev)]
				else:
					ExtItemsIntervalNew = ExtItemsInterval[(ExtItemsInterval.LSText < LSTavg + LSTstdDev) & (ExtItemsInterval.LSText > LSTavg - 2*LSTstdDev)]
				# Compute the number of remaining elements and see if any elements have been dropped during the current iteration
				Nnew = len(ExtItemsIntervalNew.index)
				if Nnew == Ncurrent or Nnew < Nmin:
					CHECK = True
				else:
					ExtItemsInterval = ExtItemsIntervalNew
					Ncurrent = Nnew
					# Since some of the items have been dropped, we need to compute a new mean and average value
					LSTavg = ExtItemsInterval.LSText.mean()
					LSTstdDev = ExtItemsInterval.LSText.std()
			# Compute the average albedo value for the remaining items
			AlbedoAvg = ExtItemsInterval.AlbedoExt.mean()
			# Now that the outliers have been eliminated, we can store the results for the current interval
			OptimalItems = pd.concat([OptimalItems, pd.DataFrame({'LSTopt':LSTavg, 'AlbedoOpt':AlbedoAvg}, index=[0])], ignore_index=True)


		### PART II ###
		# Remove items which disturb the edge detection
		# This is always the case with the dry edge, where the initial couple of M intervals will have an increasing LST value instead of a decreasing one
		# However, this can also be the case with the wet edge, where the scatter plot assumes a 'diamond shape' and the inital couple of items will have an increasing value
		# Take out the lowest range of albedo values
		PartIIData = OptimalItems[(OptimalItems.AlbedoOpt > OptimalItems.AlbedoOpt.quantile(LowestAlbedoFraction)) & (OptimalItems.AlbedoOpt < OptimalItems.AlbedoOpt.quantile(HeighestAlbedoFraction))]
		if DRY_EDGE:
			# Find the albedo of the item with the maximum LST value from the lower range of albedo values and remove all those items with a lower albedo value
			MaxLSTitem = PartIIData.loc[pd.to_numeric(PartIIData.LSTopt).idxmax()]
			OptimalItemsRevised = OptimalItems[OptimalItems.AlbedoOpt >= MaxLSTitem.AlbedoOpt]
		else:
			# Find the albedo of the item with the minimum LST value and remove all those items with a lower albedo value
			MinLSTitem = PartIIData.loc[pd.to_numeric(PartIIData.LSTopt).idxmin()]
			OptimalItemsRevised = OptimalItems[OptimalItems.AlbedoOpt >= MinLSTitem.AlbedoOpt]
		

		### PART III ###
		# Count the number of optimal items still remaining in the revised version
		Mcurrent  = len(OptimalItemsRevised.index)
		# Perform linear regression over the remaining items
		LSTvalues = np.array(OptimalItemsRevised.LSTopt)
		AlbedoValues = np.array(OptimalItemsRevised.AlbedoOpt).reshape((-1, 1))
		EDGE = LinearRegression().fit(AlbedoValues, LSTvalues)
		Prediction = EDGE.predict(AlbedoValues)
		RMSE = sqrt(mean_squared_error(y_true=LSTvalues, y_pred=Prediction))

		# While we have not yet reached the minimum allowed number of (optimal) items and it might still be possible to reduce the number of optimal items
		CHECK = False
		while Mcurrent > Mmin and CHECK == False:
			# Go through all the optimal items and see if any of them need to be dropped
			# The statement has to be true for items to be kept!
			if DRY_EDGE:
				OptimalItemsRevisedNew = OptimalItemsRevised[(OptimalItemsRevised.LSTopt >= Prediction-RSMEmultiplier*RMSE) & (OptimalItemsRevised.LSTopt<=Prediction+RSMEmultiplier*RMSE)] 
			else:
				OptimalItemsRevisedNew = OptimalItemsRevised[OptimalItemsRevised.LSTopt >= Prediction-RSMEmultiplier*RMSE] 				
			# Compute the number of remaining elements and see if any elements have been dropped during the current iteration
			Mnew = len(OptimalItemsRevisedNew.index)
			if Mnew == Mcurrent or Mnew < Mmin:
				CHECK = True
			else:
				OptimalItemsRevised = OptimalItemsRevisedNew
				Mcurrent = Mnew
				# Since some of the items have been dropped, we need to perform a new Linear Regression and compute a new RMSE
				LSTvalues = np.array(OptimalItemsRevised.LSTopt)
				AlbedoValues = np.array(OptimalItemsRevised.AlbedoOpt).reshape((-1, 1))
				EDGE = LinearRegression().fit(AlbedoValues, LSTvalues)
				Prediction = EDGE.predict(AlbedoValues)
				RMSE = sqrt(mean_squared_error(y_true=LSTvalues, y_pred=Prediction))

		# After finishing the while loop, we've found the ideal set of optimal items to perform the linear regression on
		return EDGE, OptimalItems, OptimalItemsRevised


	### PART IV ###
	# Use the compute_edge function to retrieve the dry and wet edges
	[DryEdge, DryOptItems, DryOptItemsRevised] = Compute_Edge('DRY_EDGE')
	[WetEdge, WetOptItems, WetOptItemsRevised] = Compute_Edge('WET_EDGE')

	# Next, for each of the dry and wet items in the revised set, check if it falls outside of the bounds of the opposite edge
	# If a 'wet  item' falls above the dry edge it should be removed
	# If a 'dry item' falls above the wet edge it should be remove
	# Add values to the beginning and end of the array of albedo values
	# Produce predictions
	#PredictDryItemsWithWetEdge = WetEdge.predict(np.array(DryOptItemsRevised.AlbedoOpt).reshape((-1, 1)))
	#PredictWetItemsWithDryEdge = DryEdge.predict(np.array(WetOptItemsRevised.AlbedoOpt).reshape((-1, 1)))
	# Filter out the items which fall outside the bounds of the 'other' edge
	#DryOptItemsRevised = DryOptItemsRevised[DryOptItemsRevised.LSTopt > PredictDryItemsWithWetEdge]
	#WetOptItemsRevised = WetOptItemsRevised[WetOptItemsRevised.LSTopt < PredictWetItemsWithDryEdge]


	### CONCLUSION ###
	# The optimal sets have now been found. The final dry and wet edges can now be derived using linear regression.
	LSTvalues = np.array(DryOptItemsRevised.LSTopt)
	AlbedoValues = np.array(DryOptItemsRevised.AlbedoOpt).reshape((-1, 1))
	DryEdge = LinearRegression().fit(AlbedoValues, LSTvalues)
	DryEdge_a = DryEdge.coef_[0]
	DryEdge_b = DryEdge.intercept_
	DryEdge_score = DryEdge.score(np.array(DryOptItemsRevised.AlbedoOpt).reshape((-1, 1)), np.array(DryOptItemsRevised.LSTopt))

	LSTvalues = np.array(WetOptItemsRevised.LSTopt)
	AlbedoValues = np.array(WetOptItemsRevised.AlbedoOpt).reshape((-1, 1))
	WetEdge = LinearRegression().fit(AlbedoValues, LSTvalues)
	WetEdge_a = WetEdge.coef_[0]
	WetEdge_b = WetEdge.intercept_
	WetEdge_score = WetEdge.score(np.array(WetOptItemsRevised.AlbedoOpt).reshape((-1, 1)), np.array(WetOptItemsRevised.LSTopt))


	### DISPLAY RESULTS ###
    # Report the results to the user if so required 
	if DisplayResults:
		print('\n      Results of Dry Edge:')
		print('      Coefficient of determination:', DryEdge_score)
		print('      Slope coefficient:', DryEdge_a)
		print('      Intercept coefficient:', DryEdge_b)

		print('\n      Results of Wet Edge:')
		print('      Coefficient of determination:', WetEdge_score)
		print('      Slope coefficient:', WetEdge_a)
		print('      Intercept coefficient:', WetEdge_b, '\n')

		# Create a range of albedo values
		StartAlbedo = DATA.Albedo.quantile(AlbedoQuantile)
		StopAlbedo = DATA.Albedo.quantile(1-AlbedoQuantile)

		# Add values to the beginning and end of the array of albedo values
		#DryAlbedoPrediction = np.append(np.append(DryOptItemsRevised.AlbedoOpt.iloc[0]-0.05, DryOptItemsRevised.AlbedoOpt), DryOptItemsRevised.AlbedoOpt.iloc[-1]+0.05).reshape((-1, 1))
		DryAlbedoPrediction = np.append(np.append(StartAlbedo, DryOptItemsRevised.AlbedoOpt), StopAlbedo).reshape((-1, 1))
		# Produce predictions
		DryPrediction = DryEdge.predict(DryAlbedoPrediction)
		# Add values to the beginning and end of the array of albedo values
		WetAlbedoPrediction = np.append(np.append(StartAlbedo, WetOptItemsRevised.AlbedoOpt), StopAlbedo).reshape((-1, 1))
		# Produce predictions
		WetPrediction = WetEdge.predict(WetAlbedoPrediction)

		# Create median line through scatter plot
		IntervalSize = 0.01
		AlbedoIntervals = np.arange(StartAlbedo, StopAlbedo, IntervalSize)
		MeanAlbedos = np.array([])
		MedianLSTs = np.array([])
		for Albedo_interval in AlbedoIntervals:
			# Select the data within the current interval
			DATAinterval = DATA[(DATA.Albedo >= Albedo_interval) & (DATA.Albedo < Albedo_interval+IntervalSize)]
			Albedo = Albedo_interval + (IntervalSize/2)
			MeanAlbedos = np.append(MeanAlbedos, Albedo)
			if len(DATAinterval.index) == 0:
				MedianLSTs = np.append(MedianLSTs, np.nan)
			else:
				MedianLSTs = np.append(MedianLSTs, np.median(DATAinterval.LST))

		# Visualize the results on a scatter plot
		plt.scatter(DATA.Albedo, DATA.LST, c='blue', s = 10, label="Data point")
		plt.scatter(DryOptItems.AlbedoOpt, DryOptItems.LSTopt, c='red', s = 50)
		plt.scatter(DryOptItemsRevised.AlbedoOpt, DryOptItemsRevised.LSTopt, c='black', s = 15)
		plt.plot(DryAlbedoPrediction, DryPrediction,color='red',linestyle='dashed',linewidth=2, label="Dry edge")
		plt.scatter(WetOptItems.AlbedoOpt, WetOptItems.LSTopt, c='green', s = 50)
		plt.scatter(WetOptItemsRevised.AlbedoOpt, WetOptItemsRevised.LSTopt, c='black', s = 15)
		plt.plot(WetAlbedoPrediction, WetPrediction,color='green',linestyle='dashed',linewidth=2, label="Wet edge")
		plt.legend(loc="upper right", prop={'size': 15})
		plt.tick_params(axis='x', labelsize=15)
		plt.tick_params(axis='y', labelsize=15)
		plt.title('Scatter plot with automated edge detection', fontsize=30)
		plt.xlabel('Albedo', fontsize=25)
		plt.ylabel('Land surface temperature [K]', fontsize=25)
		plt.show()	

	return [DryEdge_a, DryEdge_b, DryEdge_score, WetEdge_a, WetEdge_b, WetEdge_score]



def SPLIT_Method(DATA, AlbedoStep, AlbedoQuantile, UpperQuantile, LowerQuantile, DisplayResults=False): 
    """Def: Compute the coefficients of the evaporative fraction (EF) using the SPLIT method.
	Args:
		DATA []: 
        AlbedoStep []: 
        AlbedoQuantile []: 
        UpperQuantile []: 
        LowerQuantile []: 
    Returns:
		...
	Example:
		..."""    

    ### INITIALIZATION ###
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import numpy as np

	# Find the maximum albedo value. This will be the end of the spectrum of albedo values which we'll inspect
    StartAlbedo = DATA.Albedo.quantile(AlbedoQuantile)
    StopAlbedo = DATA.Albedo.quantile(1-AlbedoQuantile)   

	### COLLECT MAX LST VALUES ###
	# Create list of top 1% of LST values
    MaxLSTs = DATA[DATA.LST >= DATA.LST.quantile(0.99)]
	# From the list of top 1% LST values, find the minimum albedo value
    AlbdOfMaxLST = DATA.Albedo.iloc[MaxLSTs.idxmin().Albedo]
	# Create a range of albedo values which will be inspected
    RangeOfAlb = np.arange(AlbdOfMaxLST, StopAlbedo, AlbedoStep)
    MeanMaxValues = np.array([])
    MaxAlbedoValues = np.array([])
    for alb_current in RangeOfAlb:
        DATAsubset = DATA[(DATA.Albedo >= alb_current) & (DATA.Albedo < alb_current+AlbedoStep)]
		# We only need LST values now
        DATAsubset = DATAsubset.LST
		# Find maximum values
        top_percentile = DATAsubset.quantile(UpperQuantile)
        MaxLSTValues = DATAsubset[DATAsubset >= top_percentile]
		# Add values. Only do this if there is actually a value
        if len(MaxLSTValues):
            MeanMaxLST = MaxLSTValues.mean()
            Albedo_value = np.mean([alb_current, alb_current+AlbedoStep])
            MaxAlbedoValues = np.append(MaxAlbedoValues, Albedo_value)
            MeanMaxValues = np.append(MeanMaxValues, MeanMaxLST)
    MaxAlbedoValues = MaxAlbedoValues.reshape((-1, 1))
    DryEdge = LinearRegression().fit(MaxAlbedoValues, MeanMaxValues)
    DryEdge_a = DryEdge.coef_[0]
    DryEdge_b = DryEdge.intercept_
	

	### COLLECT MIN LST VALUES ###
	# Create list of bottom 1% of LST values
    MinLSTs = DATA[DATA.LST >= DATA.LST.quantile(0.01)]
	# From the list of bottom 1% LST values, find the minimum albedo value
    AlbdOfMinLST = DATA.Albedo.iloc[MinLSTs.idxmin().Albedo]
	# Create a range of albedo values which will be inspected
    RangeOfAlb = np.arange(AlbdOfMinLST, StopAlbedo, AlbedoStep)
    MeanMinValues = np.array([])
    MinAlbedoValues = np.array([])

    for alb_current in RangeOfAlb:
        # Select the subsection corresponding with the current section of albedo values 
        DATAsubset = DATA[(DATA.Albedo >= alb_current) & (DATA.Albedo < alb_current+AlbedoStep)]
        # We only need LST values now
        DATAsubset = DATAsubset.LST
		# Find minimum values
        bottom_percentile = DATAsubset.quantile(LowerQuantile)
        MinLSTValues = DATAsubset[DATAsubset <= bottom_percentile]
		# Add values. Only do this if there is actually a value
        if len(MinLSTValues) > 0:
            MeanMinLST = MinLSTValues.mean()
            Albedo_value = np.mean([alb_current, alb_current+AlbedoStep])
            MinAlbedoValues = np.append(MinAlbedoValues, Albedo_value)
            MeanMinValues = np.append(MeanMinValues, MeanMinLST)
    MinAlbedoValues = MinAlbedoValues.reshape((-1, 1))
    WetEdge = LinearRegression().fit(MinAlbedoValues, MeanMinValues)
    WetEdge_a = WetEdge.coef_[0]
    WetEdge_b = WetEdge.intercept_

    ### DISPLAY RESULTS ###
    # Report the results to the user if so required 
    if DisplayResults:
        DryEdge_score = DryEdge.score(MaxAlbedoValues, MeanMaxValues)
        WetEdge_score = WetEdge.score(MinAlbedoValues, MeanMinValues)
        print('Results of Dry Edge:')
        print('coefficient of determination:', DryEdge_score)
        print('slope:', DryEdge_a)
        print('intercept:', DryEdge_b, '\n')
        print('Results of Wet Edge:')
        print('coefficient of determination:', WetEdge_score)
        print('slope:', WetEdge_a)
        print('intercept:', WetEdge_b, '\n')

        print(StartAlbedo, StopAlbedo)

		### PRODUCE PREDICTIONS ###
        MaxAlbedoValuesPrediction = np.append(np.append(StartAlbedo, MaxAlbedoValues), StopAlbedo[-1]+0.1).reshape((-1, 1))
        MinAlbedoValuesPrediction = np.append(np.append(StartAlbedo, MinAlbedoValues), StopAlbedo[-1]+0.1).reshape((-1, 1))

        DryResults = DryEdge.predict(MaxAlbedoValuesPrediction)
        WetResults = WetEdge.predict(MinAlbedoValuesPrediction) 
		
		### PLOT ###
        plt.scatter(DATA.Albedo, DATA.LST, c='blue', s = 10)
        plt.scatter(MaxAlbedoValues, MeanMaxValues, c='red', s = 30)
        plt.plot(MaxAlbedoValuesPrediction, DryResults,color='red',linestyle='dashed',linewidth=2)
        plt.scatter(MinAlbedoValues, MeanMinValues, c='green', s = 30)
        plt.plot(MinAlbedoValuesPrediction, WetResults, color='green',linestyle='dashed',linewidth=2)
        plt.title('SPLIT METHOD: Scatter plot Albedo - Landsurface temperature')
        plt.xlabel('Albedo')
        plt.ylabel('LST (K)')
        plt.show()

    return [DryEdge_a, DryEdge_b, WetEdge_a, WetEdge_b]

