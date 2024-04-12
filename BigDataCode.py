import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import dask.dataframe as dd
from dask.distributed import Client
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#loading the datasets
trips_by_distance_df = pd.read_csv('Trips_by_Distance.csv')
trips_full_data_df = pd.read_csv('trips_full_data.csv')

#displaying the first couple of rows of both datasets
print(trips_by_distance_df.head())
print(trips_full_data_df.head())

#summary stats and info
print(trips_by_distance_df.describe())
print(trips_full_data_df.describe())

print(trips_by_distance_df.info())
print(trips_full_data_df.info())

#converting 'Date' columns to datetime format and extracting 'Week of Date' and 'Month of Date'
trips_full_data_df['Date'] = pd.to_datetime(trips_full_data_df['Date'])
trips_by_distance_df['Date'] = pd.to_datetime(trips_by_distance_df['Date'])
trips_full_data_df['Week of Date'] = trips_full_data_df['Date'].dt.isocalendar().week
trips_full_data_df['Month of Date'] = trips_full_data_df['Date'].dt.month
trips_by_distance_df['Week of Date'] = trips_by_distance_df['Date'].dt.isocalendar().week
trips_by_distance_df['Month of Date'] = trips_by_distance_df['Date'].dt.month

#making sure 'Week of Date' and 'Month of Date' columns exist in the datasets, and if not extract them from 'Date'
if 'Week of Date' not in trips_by_distance_df.columns:
    trips_by_distance_df['Week of Date'] = trips_by_distance_df['Date'].dt.isocalendar().week
    trips_full_data_df['Week of Date'] = trips_full_data_df['Date'].dt.isocalendar().week

if 'Month of Date' not in trips_by_distance_df.columns:
    trips_by_distance_df['Month of Date'] = trips_by_distance_df['Date'].dt.month
    trips_full_data_df['Month of Date'] = trips_full_data_df['Date'].dt.month

#calculating the average number of people staying at home per week (mean)
avg_staying_home_per_week = trips_by_distance_df.groupby('Week')['Population Staying at Home'].mean()

#to visualize the above
avg_staying_home_per_week.plot(kind='bar')

#grouping data by week and calculating the mean for trips withtin 1-25 miles
avg_trips_1_25_per_week = trips_full_data_df.groupby('Week of Date')['Trips 1-25 Miles'].mean()

#plotting the results
plt.figure(figsize=(12, 6))
avg_trips_1_25_per_week.plot(kind='bar', color='skyblue')
plt.title('Average number of Trips (1-25 Miles) per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=0)
plt.show()

#grouping data by week and calculating the mean for trips withtin 25-50 miles
avg_trips_25_50_per_week = trips_full_data_df.groupby('Week of Date')['Trips 25-50 Miles'].mean()

#plotting the results
plt.figure(figsize=(12, 6))
avg_trips_25_50_per_week.plot(kind='bar', color='skyblue')
plt.title('Average number of Trips (25-50 Miles) per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=0)
plt.show()

#grouping data by week and calculating the mean for trips withtin 50-100 miles
avg_trips_50_100_per_week = trips_full_data_df.groupby('Week of Date')['Trips 50-100 Miles'].mean()

#plotting the results
plt.figure(figsize=(12, 6))
avg_trips_50_100_per_week.plot(kind='bar', color='skyblue')
plt.title('Average number of Trips (50-100 Miles) per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=0)
plt.show()

#grouping data by week and calculating the mean for trips withtin 100-250 miles
avg_trips_100_250_per_week = trips_full_data_df.groupby('Week of Date')['Trips 100-250 Miles'].mean()

#plotting the results
plt.figure(figsize=(12, 6))
avg_trips_100_250_per_week.plot(kind='bar', color='skyblue')
plt.title('Average number of Trips (100-250 Miles) per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=0)
plt.show()

#grouping data by week and calculating the mean for trips withtin 100-250 miles
avg_trips_100_250_per_week = trips_full_data_df.groupby('Week of Date')['Trips 100-250 Miles'].mean()

#plotting the results
plt.figure(figsize=(12, 6))
avg_trips_100_250_per_week.plot(kind='bar', color='skyblue')
plt.title('Average number of Trips (100-250 Miles) per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=0)
plt.show()

#grouping data by week and calculating the mean for trips withtin 250-500 miles
avg_trips_250_500_per_week = trips_full_data_df.groupby('Week of Date')['Trips 250-500 Miles'].mean()

#plotting the results
plt.figure(figsize=(12, 6))
avg_trips_250_500_per_week.plot(kind='bar', color='skyblue')
plt.title('Average number of Trips (250-500 Miles) per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=0)
plt.show()

#grouping data by week and calculating the mean for trips 500+ miles
avg_trips_500p_per_week = trips_full_data_df.groupby('Week of Date')['Trips 500+ Miles'].mean()

#plotting the results
plt.figure(figsize=(6, 3))
avg_trips_500p_per_week.plot(kind='bar', color='skyblue')
plt.title('Average number of Trips (500+ Miles) per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.xticks(rotation=0)
plt.show()

#aggregating multiple distance categories
distance_columns = ['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100-250 Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles']
weekly_distance_means = trips_full_data_df.groupby('Week of Date')[distance_columns].mean()

#plotting
weekly_distance_means.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('Average Number of Trips per Distance Category Per Week')
plt.xlabel('Week')
plt.ylabel('Average Number of Trips')
plt.legend(title='Distance Categories')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#making sure 'Date' is on datetime format for plotting
trips_by_distance_df['Date'] = pd.to_datetime(trips_by_distance_df['Date'])

#filter the dataset for trips within 10-25 miles
trips_by_distance_df_10_25 = trips_by_distance_df[trips_by_distance_df['Number of Trips 10-25'] > 10000000]

#same for 50-100 miles
trips_by_distance_df_50_100 = trips_by_distance_df[trips_by_distance_df['Number of Trips 50-100'] > 10000000]

#scatter plot for the 10-25 miles range 
plt.figure(figsize=(12,6))
plt.scatter(trips_by_distance_df_10_25['Date'], trips_by_distance_df_10_25['Number of Trips 10-25'], color = 'blue', label = 'Trips 10-25 Miles')
plt.title('Number of Trips (10-25 Miles) Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.xticks(rotation=0)
plt.legend()
plt.show()

#scatter plot for the 50-100 miles range 
plt.figure(figsize=(12, 6))
plt.scatter(trips_by_distance_df_50_100['Date'], trips_by_distance_df_50_100['Number of Trips 50-100'], color='red', label='Trips 50-100 Miles')
plt.title('Number of Trips (50-100 Miles) Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.xticks(rotation=0)
plt.legend()
plt.show()

#file path
file_path = 'Trips_By_Distance.csv'

#initializing dictionary to store time taken
processing_time = {}

#start timer
start_time = time.time()

#loading data set
trips_by_distance_df = pd.read_csv(file_path)

#converting 'Date' into datetime format
trips_by_distance_df['Date'] = pd.to_datetime(trips_by_distance_df['Date'])
    
#count trips within 1-25 miles for more than 10,000,000     
trips_by_distance_df_filtered_10_25 = trips_by_distance_df[trips_by_distance_df['Number of Trips 10-25'] > 10000000]
count_10_25 = trips_by_distance_df_filtered_10_25['Number of Trips 10-25'].count()

#same for 50-100 miles 
trips_by_distance_df_filtered_50_100 = trips_by_distance_df[trips_by_distance_df['Number of Trips 50-100'] > 10000000]
count_50_100 = trips_by_distance_df_filtered_50_100['Number of Trips 50-100'].count()

#end timer and store time taken
pandas_time = time.time() - start_time
processing_time['pandas'] = pandas_time

#printing out computation times
print(f"Time taken: {pandas_time} seconds")   
#filtering both datasets for Week 32
trips_full_data_df_week32 = trips_full_data_df[trips_full_data_df['Week of Date'] == 'Week 32']
trips_by_distance_df_week32 = trips_by_distance_df[trips_by_distance_df['Week'] == 32]

#merge datasets on 'Date'
merged_df = pd.merge(trips_full_data_df_week32, trips_by_distance_df_week32, on='Date')

#features and target selection
X = merged_df[['Trips 1-25 Miles', 'Trips 25-100 Miles']]
y = merged_df['Number of Trips 5-10'] + merged_df['Number of Trips 10-25']

#make sure we dont get any empty dataframes
print("Shape of merged dataframe: ", merged_df.shape)
print("Shape of x: ", X.shape)
print("Shape of y: ", y.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#converting 'Date' columns to datetime format in both dataframes
trips_full_data_df['Date'] = pd.to_datetime(trips_full_data_df['Date'])
trips_by_distance_df['Date'] = pd.to_datetime(trips_by_distance_df['Date'])

#merge dataframes on 'Date'
merged_df = pd.merge(trips_full_data_df, trips_by_distance_df, on='Date', suffixes=('_full', '_distance'))

#removing any rows with NaN values in the newly merged dataframe
merged_df.dropna(inplace=True)

#'Population Not Staying at Home' is target
#all other columns except 'Date' are features
X = merged_df.drop(['Date', 'Population Not Staying at Home'], axis=1)
y = merged_df['Population Not Staying at Home']

#converting categorical columns to numeric
X = pd.get_dummies(X)

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initializing and training the Linear Regression model
model_1 = LinearRegression()
model_1.fit(X_train, y_train)

#Linear Rgeression Model Evaluation
from sklearn.metrics import mean_squared_error, r2_score

y_pred_1 = model_1.predict(X_test)
mse_1 = mean_squared_error(y_test, y_pred_1)
r2_1 = r2_score(y_test, y_pred_1)

print(f"MSE: {mse_1}, R^2: {r2_1}")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#initializing and training the Polynomial Regression Model
degree = 1
model_2 = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_2.fit(X_train, y_train)

#Polynomial Regression Model Evaluation
y_pred_2 = model_2.predict(X_test)
mse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)

print(f"MSE: {mse_2}, R^2: {r2_2}")

#displaying shapes of splits to confirm the operation
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#define categories of trips
distance_categories = [
    'Trips 1-25 Miles', 'Trips 25-50 Miles', 'Trips 50-100 Miles',
    'Trips 100-250 Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles'
]

#summing number of travelers for each distance category
distance_data = trips_full_data_df[distance_categories].multiply(trips_full_data_df['People Not Staying at Home'], axis="index").sum()

#plotting travelers by distance categories
plt.figure(figsize=(12, 6))
sns.barplot(x=distance_data.index, y=distance_data.values, palette='viridis')
plt.title('Number of Travelers by Distance Categories')
plt.xlabel('Distance Categories')
plt.ylabel('Number of Travelers')
plt.xticks(rotation=45)
plt.show()

#creating dataframe for plotting
plot_data = trips_full_data_df[['People Not Staying at Home', 'Trips']].groupby('Trips').sum().reset_index()


#plotting travelers by number of trips
plt.figure(figsize=(12, 6))
sns.barplot(x='Trips', y='People Not Staying at Home', data=plot_data, palette='magma')
plt.title('Number of Travelers by Number of Trips')
plt.xlabel('Number of Trips')
plt.ylabel('Number of Travelers')
plt.xticks(rotation=0)
plt.show()
