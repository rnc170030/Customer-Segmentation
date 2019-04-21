#cohort analysis -
#Cohorts - Mutually exclusive events
#Customer and product cohorts

#Elements of Cohort Analysis
#Pivot Table
#Assign cohort in rows
#Assign index in columns

#cohort can be time or behavior

#time cohorts - 
#Group customers baed on time when they first completed their activity

#Key Steps of Segmentation Project -
#1 - Gaher data, updated data with an additional variables - Recency, Frequency, Monetary.
#2 - PreProcess the data
#3 - Explore the data and decide on no. of clusters
#4 - Run Kmeans clustering
#5 - Analyze and visualize results
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import StandardScaler

online = pd.read_csv('online12M.csv');

# Define a function that will parse the date
def get_day(x): return dt.date(x.year, x.month, x.day) 

# Create InvoiceDay column
online['InvoiceDate'] = pd.to_datetime(online['InvoiceDate'])
online['InvoiceDay'] = online['InvoiceDate'].apply(get_day) 

# Group by CustomerID and select the InvoiceDay value
grouping = online.groupby('CustomerID')['InvoiceDay'] 

# Assign a minimum InvoiceDay value to the dataset
online['CohortDay'] = grouping.transform('min')

# View the top 5 rows
print(online.head())

#First, we will create 6 variables that capture the integer value of years, months and days for Invoice and Cohort Date using the get_date_int() function that's been already defined
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day

online['InvoiceDay'] = pd.to_datetime(online['InvoiceDay'])
online['CohortDay'] = pd.to_datetime(online['CohortDay'])

# Get the integers for date parts from the `InvoiceDay` column
online['invoice_year'], online['invoice_month'], online['invoice_day'] = get_date_int(online, 'InvoiceDay')

# Get the integers for date parts from the `CohortDay` column
online['cohort_year'], online['cohort_month'], online['cohort_day'] = get_date_int(online, 'CohortDay')

# Calculate difference in years
years_diff =online['invoice_year'] - online['cohort_year']

# Calculate difference in months
months_diff = online['invoice_month'] - online['cohort_month']

# Calculate difference in days
days_diff = online['invoice_day'] - online['cohort_day']

# Extract the difference in days from all previous values
online['CohortIndex'] = years_diff * 365 + months_diff * 30 + days_diff + 1
print(online.head())


grouping = online.groupby(['cohort_month', 'CohortIndex'])

# Count the number of unique values per customer ID
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique).reset_index()

# Create a pivot 
cohort_counts = cohort_data.pivot(index='cohort_month', columns='CohortIndex', values='CustomerID')

# Select the first column and store it to cohort_sizes
cohort_sizes = cohort_counts.iloc[:,0]

# Divide the cohort count by cohort sizes along the rows
retention = cohort_counts.divide(cohort_sizes, axis=0)

# Create a groupby object and pass the monthly cohort and cohort index as a list
grouping = online.groupby(['cohort_month', 'CohortIndex']) 

# Calculate the average of the unit price column
cohort_data = grouping['UnitPrice'].mean()

# Reset the index of cohort_data
cohort_data = cohort_data.reset_index()

# Create a pivot 
average_quantity = cohort_data.pivot(index='cohort_month', columns='CohortIndex', values='UnitPrice')
print(average_quantity.round(1))

# Import seaborn package as sns
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize an 8 by 6 inches plot figure
#plt.figure(figsize=(8, 6))

# Add a title
plt.title('Average Spend by Monthly Cohorts')

# Create the heatmap
sns.heatmap(data = average_quantity, annot=True, cmap='Blues')
plt.show()


#Receny, Frequency, Monetary Segmentation - RFM
#Recency - how recent was each cstomer's last purchase
#frequency - how many purchases customer has done in last 12 months
#monetary - how much has customer spend in last 12 months

#RFM values can be grouped in Percentiles, hig/low values split, custom - based on business knowledge
#data = pd.read_csv('datamart_rfm.csv')
# Create a spend quartile with 4 groups - a range between 1 and 5
#spend_quartile = pd.qcut(data['Spend'], q=4, labels=range(1,5))

# Assign the quartile values to the Spend_Quartile column in data
#data['Spend_Quartile'] = spend_quartile

# Print data with sorted Spend values
#print(data.sort_values('Spend'))

# Store labels from 4 to 1 in a decreasing order
#r_labels = list(range(4, 0, -1))

# Create a spend quartile with 4 groups and pass the previously created labels 
#recency_quartiles = pd.qcut(data['Recency_Days'], q=4, labels=r_labels)

# Assign the quartile values to the Recency_Quartile column in `data`
#data['Recency_Quartile'] = recency_quartiles 

# Print `data` with sorted Recency_Days values
#print(data.sort_values('Recency_Days'))


print('Min:{}; Max{}'.format(min(online.InvoiceDate), max(online.InvoiceDate)))
online['TotalSum'] = online['Quantity'] * online['UnitPrice']

snapshot_date = max(online.InvoiceDate) + dt.timedelta(days=1)
# Calculate Recency, Frequency and Monetary value for each customer 
datamart = online.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Rename the columns 
datamart.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue'}, inplace=True)

# Print top 5 rows
print(datamart.head())

#            Recency  Frequency  MonetaryValue
#CustomerID                                   
#12747             3         25         948.70
#12748             1        888        7046.16
#12749             4         37         813.45
#12820             4         17         268.02
#12822            71          9         146.15

# Create labels for Recency and Frequency
r_labels = range(3, 0, -1); f_labels = range(1, 4)

# Assign these labels to three equal percentile groups 
r_groups = pd.qcut(datamart['Recency'], q=3, labels=r_labels)

# Assign these labels to three equal percentile groups 
f_groups = pd.qcut(datamart['Frequency'], q=3, labels=f_labels)

# Create new columns R and F 
datamart = datamart.assign(R= r_groups.values, F=f_groups.values)

# Create labels for MonetaryValue 
m_labels = range(1, 4)

# Assign these labels to three equal percentile groups
m_groups = pd.qcut(datamart['MonetaryValue'], q=3, labels=m_labels)

# Create new column M
datamart = datamart.assign(M=m_groups.values)

# Calculate RFM_Score
datamart.index = datamart.set_index(['R','F','M'])
datamart['RFM_Score'] = datamart[['R','F','M']].sum(axis=1)
print(datamart['RFM_Score'].head())

# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 10:
        return 'Top'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 10)):
        return 'Middle'
    else:
        return 'Low'

# Create a new variable RFM_Level
datamart['RFM_Level'] = datamart.apply(rfm_level, axis=1)

# Print the header with the top 5 rows to the console.
print(datamart.head())

# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = datamart.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
  
  	# Return the size of each segment
    'MonetaryValue': ['mean', 'count']
}).round(1)

# Print the aggregated dataset
print(rfm_level_agg)

#              Recency Frequency MonetaryValue      
#                mean      mean          mean count
#    RFM_Level                                      
#    Low         180.8       3.2          52.7  1075
#    Middle       73.9      10.7         202.9  1547
#    Top          20.3      47.1         959.7  1021

#K means clustering
# Plot distribution of var1
plt.subplot(3, 1, 1); sns.distplot(data['var1'])

# Plot distribution of var2
plt.subplot(3, 1, 2); sns.distplot(data['var2'])

# Plot distribution of var3
plt.subplot(3,1,3); sns.distplot(data['var3'])

# Show the plot
plt.show()

# Plot recency distribution
plt.subplot(3, 1, 1); sns.distplot(datamart['Recency'])

# Plot frequency distribution
plt.subplot(3, 1, 2); sns.distplot(datamart['Frequency'])

# Plot monetary value distribution
plt.subplot(3, 1, 3); sns.distplot(datamart['MonetaryValue'])

# Show the plot
plt.show()

# Unskew the data
datamart_log = np.log(datamart)

# Initialize a standard scaler and fit it
scaler = StandardScaler()
scaler.fit(datamart_log)

# Scale and center the data
datamart_normalized = scaler.transform(datamart_log)

# Create a pandas DataFrame
datamart_normalized = pd.DataFrame(data=datamart_normalized, index=datamart.index, columns=datamart_rfm.columns)

#Steps -
#Datapreprocessing, Choose no. of clusters, Run kmeans, Analyze average RFM values for each cluster
# Import KMeans 
from sklearn.cluster import KMeans

# Initialize KMeans
kmeans = KMeans(n_clusters=3, random_state=1) 

# Fit k-means clustering on the normalized data set
kmeans.fit(datamart_normalized)

# Extract cluster labels
cluster_labels = kmeans.labels_

# Create a DataFrame by adding a new cluster label column
datamart_rfm_k3 = datamart.assign(Cluster=cluster_labels)

# Group the data by cluster
grouped = datamart_rfm_k3.groupby(['Cluster'])

# Calculate average RFM values and segment sizes per cluster value
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
  }).round(1)
    
# Fit KMeans and calculate SSE for each k
for k in range(1, 21):
  
    # Initialize KMeans with k clusters
    kmeans = KMeans(n_clusters=k, random_state=1)
    
    # Fit KMeans on the normalized dataset
    kmeans.fit(data_normalized)
    
    # Assign sum of squared distances to k element of dictionary
    sse[k] = kmeans.inertia_ 
    

# Add the plot title "The Elbow Method"
plt.title('The Elbow Method')

# Add X-axis label "k"
plt.xlabel('k')

# Add Y-axis label "SSE"
plt.ylabel('SSE')

# Plot SSE values for each key in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

#Snake Plot to understand and compare segments
#Market research technique to compare different segments
#Visual Representation of each segment's attribute
#Need to first normalize data
#Plot each cluster's average normalized value of each attribute

# Melt the normalized dataset and reset the index
datamart_melt = pd.melt(
  					datamart_normalized.reset_index(), 
                        
# Assign CustomerID and Cluster as ID variables                  
                    id_vars=['CustomerID', 'Cluster'],

# Assign RFM values as value variables
                    value_vars=['Recency', 'Frequency', 'MonetaryValue'], 
                        
# Name the variable and value
                    var_name='Metric', value_name='Value'
					)

# Add the plot title
plt.title('Snake plot of normalized variables')

# Add the x axis label
plt.xlabel('Metric')

# Add the y axis label
plt.ylabel('Value')

# Plot a line for each value of the cluster variable
sns.lineplot(data=datamart_melt, x='Metric', y='Value', hue='Cluster')
plt.show()

# Calculate average RFM values for each cluster
cluster_avg = datamart_rfm_k3.groupby(['Cluster']).mean() 

# Calculate average RFM values for the total customer population
population_avg = datamart_rfm.mean()

# Calculate relative importance of cluster's attribute value compared to population
relative_imp = cluster_avg / population_avg - 1

# Print relative importance score rounded to 2 decimals
print(relative_imp.round(2))

#             Recency  Frequency  MonetaryValue
#    Cluster                                   
#    0           0.84      -0.84          -0.86
#    1          -0.15      -0.35          -0.42
#    2          -0.82       1.67           1.82

#Relative importance should be greater than 0.

# Initialize a plot with a figure size of 8 by 2 inches 
plt.figure(figsize=(8, 2))

# Add the plot title
plt.title('Relative importance of attributes')

# Plot the heatmap
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()

#Apart from RFM variables we can also add Tenure
#Tenure - Time since first transaction of the customer
