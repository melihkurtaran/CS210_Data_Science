# -*- coding: utf-8 -*-
"""
# Assignment Done by Melih Kurtaran

In this assignment, you are going to practice your exploratory data analysis skills. Each question is accompanied by an incomplete function. Your task is to construct the body of those functions with appropriate return types.

"""

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

"""## Exploratory Data Analysis

> Exploratory data analysis is an attitude, a state of flexibility, a willingness to look for those things that we believe are not there, as well as those we believe to be there.

John Tukey

## About the Dataset
This dataset consists of data from 1985 Ward's Automotive Yearbook. Here are the sources:

Sources:
 >1) 1985 Model Import Car and Truck Specifications, 1985 Ward's Automotive Yearbook.  
 2) Personal Auto Manuals, Insurance Services Office, 160 Water Street, New York, NY 10038  
 3) Insurance Collision Report, Insurance Institute for Highway Safety, Watergate 600, Washington, DC 20037

**Content**  
This data set consists of three types of entities:  
(a) the specification of an auto in terms of various characteristics,   
(b) its assigned insurance risk rating,  
(c) its normalized losses in use as compared to other cars. 

The second rating corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process "symboling". A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.

The third factor is the relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification (two-door small, station wagons, sports/speciality, etc...), and represents the average loss per car per year.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os.path import join

# %matplotlib inline

"""The first step is to read in the data."""

path = "/content/gdrive/My Drive/"
filename = "Automobile_data.csv"

df = pd.read_csv(join(path, filename))

"""## Data Cleaning

### Q1-Changing Column Names

Change the column name "make" to "manufacturer".
"""

def changeColumnName(df):
  newdf = df
  newdf.rename(columns={"make":"manufacturer"},inplace=True)
  return newdf

#DO NOT CHANGE OR REMOVE
df = changeColumnName(df)

"""### Q2-Consistent NaN values

Replace all column values which contain ‘?’ or n.a with Numpy's nan value, np.nan.
"""

def replaceWithNaN(df):
  newdf = df.replace("?",np.nan)
  newdf = newdf.replace("n.a",np.nan)
  return newdf

#DO NOT CHANGE OR REMOVE
df = replaceWithNaN(df)

"""### Q3-Converting to Numeric Type

Columns `"normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"` should be numeric.
"""

def makeNumericCols(df):
  newdf = df
  newdf["normalized-losses"] = pd.to_numeric(df["normalized-losses"])
  newdf["bore"] = pd.to_numeric(newdf["bore"])
  newdf["stroke"] = pd.to_numeric(newdf["stroke"])
  newdf["horsepower"] = pd.to_numeric(newdf["horsepower"])
  newdf["peak-rpm"] = pd.to_numeric(newdf["peak-rpm"])
  newdf["price"] = pd.to_numeric(newdf["price"])
  return newdf

# In case you are using other arguments in the function above, do not forget them while calling the function
#DO NOT CHANGE OR REMOVE
df = makeNumericCols(df)

"""### Q4-NaN Ratios

Find the proportion of na values present in all the columns.
"""

def checkNaProportion(df):
  r = df.shape[0]
  newdf = df.isnull().sum() / r
  return newdf.sort_values(ascending=False)
  
  
checkNaProportion(df)

"""### Q5-Fill NaN Values

Fill NaN values with the mean value of their corresponding columns.
"""

def fillNaWithMean(df):
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  newdf = df.select_dtypes(include=numerics)
  for col in newdf.columns.values:
    df[col].fillna(df[col].mean(), inplace = True) 
  return df

#DO NOT CHANGE OR REMOVE
df = fillNaWithMean(df)

"""## Global Properties

### Q1-Column Distribution


Given a column name, plot a histogram to display the distribution in that column.
"""

def plotHist(df, col, binss):
  plt.hist(df[col], bins=binss, rwidth=0.9)
  plt.xlabel(col)
  plt.ylabel("Frequency")
  plt.show()

plotHist(df, "horsepower", 20)

"""### Q2-Outliers

https://miro.medium.com/max/18000/1*2c21SkzJMf3frPXPAR_gZA.png

Given a column, display a boxplot to show the distribution in that column. Comment on your findings with respect the distribution and outliers.
"""

def boxplot(df, col):
  df.boxplot(column=[col])

"""**My Comments:** We see that hoursepower is usually between 70-120"""

boxplot(df, "horsepower")

"""### Q3-Displaying Relationships

Given two columns, display a scatter plot in which column values are placed in x and y axes.
"""

def plotScatter(df, col1, col2):
  df.plot.scatter(x=col1,y=col2)
  '''
  df: (dataframe) input dataframe 
  col1: the column name to be displayed in the x-axis
  col2: the column name to be displayed in the y-axis
  '''

plotScatter(df, "price", "horsepower")

"""### Q4-Correlation between Columns

Given a list of column names, create and plot the correlation matrix among them. In addition, write down your remarks and comments from the resulting figure.

**Hint:** You may check this (https://stackoverflow.com/questions/43021762/matplotlib-how-to-change-figsize-for-matshow) to change the size of a matshow figure.
"""

def plotCorr(df, cols):
  matfig = plt.figure(figsize=(10,10))
  plt.matshow(df[cols].corr(), fignum=matfig.number)
  cb = plt.colorbar()
  plt.clim(-1,1)
  cb.ax.tick_params(labelsize=10)
  plt.xticks(np.arange(6), df[cols])
  plt.yticks(np.arange(6), df[cols])
   
plotCorr(df, ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"])


"""My Comments: We see that horsepower is positively correlated with bore. However bore is negatively correlated with peak-rpm

Obviously, we can also see that when horsepower increases price also increases.

## Group Properties

### Q1-The Most Expensive Car

Print the most expensive for car with its name and price.
"""

def mostExpensiveCar(df):
  mostExpensiveCar = df[df['price']==df['price'].max()]
  return mostExpensiveCar[["manufacturer","price"]]


mostExpensiveCar(df)

"""### Q2-Total Production

Print the total amount of cars produced for each manufacturer.
"""

def carsPerCompany(df):
  df["Count"] = ""
  return df.groupby(df["manufacturer"])[["Count"]].count()
  
carsPerCompany(df)

"""### Q3-Max Priced Cars

Print each manufacturer’s highest price car.
"""

def highestPricedCarsByMake(df):
  car_Manufacturers = df.groupby('manufacturer')
  priceDf = car_Manufacturers[['price']].max()
  return priceDf

highestPricedCarsByMake(df)

"""### Q4-Average Horsepower

Print the average horsepower of each manufacturer.
"""

def housePowerByMake(df):
  car_Manufacturers = df.groupby('manufacturer')
  hoursePowerDf = car_Manufacturers[['horsepower']].mean()
  return hoursePowerDf

housePowerByMake(df)

"""### Q5-Risk for Each Manufacturer

Utilize a bar plot to display the average symboling (risk) for each manufacturer.
"""

def plotRisk(df, col):
  car_Manufacturers = df.groupby('manufacturer')
  riskDf = car_Manufacturers[[col]].mean()
  plt.ylabel("manufacturer")
  plt.title(col)
  bar_width=0.5
  plt.barh(riskDf.index,riskDf[col],bar_width)

  
plotRisk(df, "symboling")

