import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import PredictionErrorDisplay
import mglearn
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import durbin_watson
import statsmodels.stats.api as sms


#DATA EXPORT
flight_df_full=pd.read_csv(r"C:\Users\emers\OneDrive\Área de Trabalho\Flight_Data_Table2.csv")
flight_df_full.rename(columns={'price_transformed':'price_variation'}, inplace=True)
flight_df_full['time_lapse_adjusted']=flight_df_full['time_lapse_adjusted'].apply(lambda x: x+1)


plt.scatter(flight_df_full['time_lapse_adjusted'], flight_df_full['price_variation'], alpha=0.05)
plt.xlabel('Time Lapse')
plt.ylabel('Price Variation')
plt.title('Figure 5: Graph of the (Price Variation x Time Lapse) Relationship')
plt.show()


#DATA CLEANING
sns.boxplot(data=flight_df_full, x='time_lapse_adjusted', y='price_variation')
plt.xlabel('Time Lapse')
plt.ylabel('Price Variation')
plt.title('Figure 3: Graph of the (Boxplot x Time Lapse) Relationship')
plt.show()


flight_df_part3=pd.DataFrame()
for i in range(1,len(flight_df_full['time_lapse_adjusted'].unique())+1):
    Q1=flight_df_full.query(f'time_lapse_adjusted=={str(i)}')['price_variation'].quantile(0.25)
    Q3=flight_df_full.query(f'time_lapse_adjusted=={str(i)}')['price_variation'].quantile(0.75)
    IQR=Q3-Q1
    #filter=(flight_df_part['price_variation']>= Q1 -1.5*IQR) & (flight_df_part['price_variation']<=Q3+1.5*IQR)
    flight_df_part3=pd.concat([flight_df_part3, flight_df_full.query(f'price_variation>={Q1 -1.5*IQR} and price_variation<={Q3+1.5*IQR} and time_lapse_adjusted=={str(i)}')], ignore_index=True, axis=0)
flight_df_full=flight_df_part3


#DATA PREPROCESSING

#Correction of unbalance due to different data capture frequencies per hour
flight_df_part_gb1=flight_df_full.groupby(by=['place_destination', 'date_flight_departure','dow_flight_departure', 'time_flight_departure.hour','time_lapse_adjusted', 'dow_date_scrapping','time_scrapping.hour'])['price_variation'].mean().reset_index()
flight_df_part1=flight_df_part_gb1.loc[:,['dow_date_scrapping', 'time_scrapping.hour','dow_flight_departure','time_flight_departure.hour','price_variation', 'time_lapse_adjusted']]

#Correction for imbalance due to different dates, regardless of destination
flight_df_part=flight_df_part1.groupby(by=['dow_flight_departure', 'time_flight_departure.hour','time_lapse_adjusted', 'dow_date_scrapping','time_scrapping.hour'])['price_variation'].mean().reset_index()
flight_df_part2=flight_df_part.loc[:,:]

#Categorization between shifts
#20h left out
flight_df_part['time_scrapping.hour']=flight_df_part['time_scrapping.hour'].astype(str)
flight_df_part['time_scrapping.hour']=flight_df_part['time_scrapping.hour'].apply(lambda x: 'Morning' if x in ['6','7','8','9','10','11'] else x)
flight_df_part['time_scrapping.hour']=flight_df_part['time_scrapping.hour'].apply(lambda x: 'Afternoon' if x in ['12','13','14','15','16','17'] else x)
flight_df_part['time_scrapping.hour']=flight_df_part['time_scrapping.hour'].apply(lambda x: 'Night' if x in ['18','19','21','22','23'] else x)
flight_df_part['time_scrapping.hour']=flight_df_part['time_scrapping.hour'].apply(lambda x: 'Dawn' if x in ['0','1','2','3','4','5'] else x)


#1h left out
flight_df_part['time_flight_departure.hour']=flight_df_part['time_flight_departure.hour'].astype(str)
flight_df_part['time_flight_departure.hour']=flight_df_part['time_flight_departure.hour'].apply(lambda x: 'Morning' if x in ['6','7','8','9','10','11'] else x)
flight_df_part['time_flight_departure.hour']=flight_df_part['time_flight_departure.hour'].apply(lambda x: 'Afternoon' if x in ['12','13','14','15','16','17'] else x)
flight_df_part['time_flight_departure.hour']=flight_df_part['time_flight_departure.hour'].apply(lambda x: 'Night' if x in ['18','19','20','21','22','23'] else x)
flight_df_part['time_flight_departure.hour']=flight_df_part['time_flight_departure.hour'].apply(lambda x: 'Dawn' if x in ['0','2','3','4','5'] else x)
flight_df_part3=flight_df_part

#Correction of shift data imbalance
flight_df_part=flight_df_part.groupby(by=['dow_flight_departure', 'time_flight_departure.hour','time_lapse_adjusted', 'dow_date_scrapping','time_scrapping.hour'])['price_variation'].mean().reset_index()
flight_df_part4=flight_df_part


#DATA BALANCING ANALYSIS
'''
#Time_Lapse
plt.hist(flight_df_part['time_lapse_adjusted'], bins=flight_df_part['time_lapse_adjusted'].nunique())
plt.xlabel('time_lapse_adjusted')
plt.ylabel('Frequency')
plt.show()

#DOW
plt.hist(flight_df_part['dow_flight_departure'],  bins=flight_df_part['dow_flight_departure'].nunique())
plt.xlabel('dow_flight_departure')
plt.ylabel('Frequency')
plt.show()

#DOW
plt.hist(flight_df_part['dow_date_scrapping'],  bins=flight_df_part['dow_date_scrapping'].nunique())
plt.xlabel('dow_date_scrapping')
plt.ylabel('Frequency')
plt.show()

#HOUR
plt.hist(flight_df_part['time_flight_departure.hour'],  bins=flight_df_part['time_flight_departure.hour'].nunique())
plt.xlabel('time_flight_departure.hour')
plt.ylabel('Frequency')
plt.show()

#HOUR
plt.hist(flight_df_part['time_scrapping.hour'],  bins=flight_df_part['time_scrapping.hour'].nunique())
plt.xlabel('time_scrapping.hour')
plt.ylabel('Frequency')
plt.show()


'''
#Average Price_Variation Analysis
average_price_bytl=flight_df_part.groupby(by='time_lapse_adjusted')['price_variation'].mean()
plt.scatter(flight_df_part['time_lapse_adjusted'], flight_df_part['price_variation'], label='Price_Variation')
plt.scatter(average_price_bytl.index, average_price_bytl.values, color='red', label='Average Price Variation')
#plt.bar(average_price_bytl.index, average_price_bytl.values)
plt.xlabel('Time Lapse')
plt.ylabel('Price Variation')
plt.title("Figure 1: Graph of the (Price Variation and Average Price Variation)x(Time Lapse) Relationship", fontdict={'fontsize':'9'})
plt.legend()
plt.show()

#DATA TRANSFORM

#One_Hot_Encoding

flight_df_part['time_scrapping.hour']=flight_df_part['time_scrapping.hour'].astype(str)
flight_df_part['time_flight_departure.hour']=flight_df_part['time_flight_departure.hour'].astype(str)

categories=[('dow_date_scrapping', pd.unique(flight_df_part['dow_date_scrapping'])),
            ('time_scrapping.hour', pd.unique(flight_df_part['time_scrapping.hour'])),
            ('dow_flight_departure', pd.unique(flight_df_part['dow_flight_departure'])),
            ('time_flight_departure.hour', pd.unique(flight_df_part['time_flight_departure.hour']))
            ]

ohe_columns=[x[0] for x in categories]
ohe_categories=[x[1] for x in categories]

encoder=OneHotEncoder(sparse_output=False, categories=ohe_categories)
transformer=make_column_transformer((encoder, ohe_columns), remainder='passthrough')

transformed=transformer.fit_transform(flight_df_part)

flight_df_part=pd.DataFrame(transformed, columns=transformer.get_feature_names_out(), index=flight_df_part.index)

#Column Renaming
flight_df_part.columns=[x[len('onehotencoder__'):] for x in flight_df_part.columns if 'onehotencoder__' in x] + [x[len('remainder__'):] for x in flight_df_part.columns if 'remainder__' in x]

#Deleting columns to remove multicollinarity from categorical variables
flight_df_part=flight_df_part.drop(['dow_flight_departure_Monday','dow_date_scrapping_Thursday', 'time_scrapping.hour_20','time_flight_departure.hour_1'], axis=1)
flight_df_part5=flight_df_part.loc[:,:]

#Logarithmic transformation
flight_df_part['time_lapse_adjusted']=np.log10(flight_df_part['time_lapse_adjusted'])

#Centralization to remove multicollinarity from numerical variables
mean_X_x1=np.mean(flight_df_part['time_lapse_adjusted'])
flight_df_part['time_lapse_adjusted']=flight_df_part['time_lapse_adjusted']-mean_X_x1

#DATA SPLIT
X=np.transpose([flight_df_part[x].to_numpy() for x in flight_df_part.columns if x not in ['dow_date_scrapping', 'time_scrapping.hour', 'price_variation', 'dow_flight_departure', 'time_flight_departure.hour', 'sd_by_time_lapse']])

#Addition of Interaction term
X=np.hstack([X,X[:,:21]*np.transpose([flight_df_part['time_lapse_adjusted'].to_numpy()])])
#deleting the time_lapse variable, but keeping time_lapse-squared
X=np.delete(X,(20), axis=1)
y=flight_df_part['price_variation'].to_numpy()

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=0)



#ORDINARY LEAST SQUARES
print('Regression without scaling')
lr=LinearRegression(n_jobs=-1)
#lr.fit(X_trainval, y_trainval)
#lr.fit(X, y, sample_weight=np_weight)
lr.fit(X, y)
#scores=cross_val_score(lr,X, y, cv=5)
#print('Accuracy of Linear Regression Model')
#print('Training set score: ', lr.score(X_trainval, y_trainval))
#print('Test set score: ', np.mean(scores))
print('R² score: ', lr.score(X, y))
print("Linear Regression Model Coefficents: ", lr.coef_)
print("Linear Regression Model Independent Term: ", lr.intercept_, '\n')
#print("Predições: ", lr.predict(X_trainval))



plt.scatter(10**(flight_df_part['time_lapse_adjusted']), flight_df_part['price_variation'], label='Observed')
plt.scatter(10**(flight_df_part['time_lapse_adjusted']),lr.predict(X), label='Predict', color='red')
plt.show()

#RESIDUAL ANALYSIS OF OLS
residual_graph=PredictionErrorDisplay.from_estimator(lr, X,y, subsample=100000)
residuals=residual_graph.y_true-residual_graph.y_pred
residuals_1=residuals
plt.show()

plt.scatter(residual_graph.y_pred,residuals, label='Predict', alpha=0.1)
plt.xlabel('Predicted Values')
plt.ylabel('Residual')
plt.title('Figure 4: Graph of the (Residual x Predicted Values) Relationship')
plt.show()

'''
plt.scatter(residual_graph.y_true,residuals, label='Predict', alpha=0.1)
plt.xlabel('True Values')
plt.ylabel('Custom Residual')
plt.show()
'''


plt.scatter(10**(flight_df_part['time_lapse_adjusted']+mean_X_x1), residuals, alpha=0.1)
plt.ylabel('Residuals')
plt.xlabel('Time Lapse')
plt.title('Figure 6: Graph of the (Residual x Time Lapse) Relationship')
plt.show()


plt.hist(residuals, bins=100)
plt.axvline(x=np.mean(residuals), color='green')
plt.ylabel('Frequency')
plt.xlabel('Residuals Hist')
plt.show()

sns.histplot(residuals, bins=100, stat='density', kde=True)
plt.show()

#Using the Statsmodel library to generate summary statistical parameters
print('Summary')
X=sm.add_constant(X.astype(float))
results=sm.OLS(y.astype(float), X.astype(float)).fit()
print(results.summary())

#WEIGHTED LINEAR REGRESSION

R_Cov_M=np.diag((10**(flight_df_part['time_lapse_adjusted']))**2)
R_Cov_D_M=np.diag(R_Cov_M)

#WLS in Scikit Learn
print('WEIGHTED LINEAR REGRESSION')
#ORDINARY LEAST SQUARES
lr=LinearRegression(n_jobs=-1)
#lr.fit(X_trainval, y_trainval)
lr.fit(X, y, sample_weight=R_Cov_D_M)
#scores=cross_val_score(lr,X, y, cv=5)
#print('Accuracy of Linear Regression Model')
#print('Training set score: ', lr.score(X_trainval, y_trainval))
#print('Test set score: ', np.mean(scores))
print('R² score: ', lr.score(X, y))
print("Linear Regression Model Coefficents: ", lr.coef_)
print("Linear Regression Model Independent Term: ", lr.intercept_, '\n')
#print("Predições: ", lr.predict(X_trainval))


'''
residual_graph=PredictionErrorDisplay.from_estimator(lr, X,y, subsample=100000)
residuals=residual_graph.y_true-residual_graph.y_pred
residuals_2=residuals
plt.show()

plt.scatter(residual_graph.y_pred , residuals)
plt.ylabel('Standartization Residuals')
plt.show()


plt.hist(residuals, bins=100)
plt.axvline(x=np.mean(residuals), color='green')
plt.ylabel('Frequency')
plt.xlabel('Residuals')
plt.show()

plt.scatter(flight_df_part['time_lapse_adjusted'], residuals, alpha=0.2)
plt.ylabel('Residuals')
plt.xlabel('time_lapse_adjusted')
plt.show()
'''



#QQPlot
'''
stats.probplot(residuals.astype(float), dist='norm', plot=plt)
plt.show()

'''
#Summary of WLS in Statsmodel
X=sm.add_constant(X.astype(float))
WLS=sm.WLS(y.astype(float), X.astype(float), weights=R_Cov_D_M)
results=sm.OLS(WLS.wendog,WLS.wexog).fit()
predictions1=results.predict(X)
predictions2=results.predict(WLS.wexog)
r_squared=results.rsquared
print(results.summary())

#Residual Studentized
OLS_Influence=OLSInfluence(results)
residual_studentized=OLS_Influence.resid_studentized
d_w2=durbin_watson(residual_studentized)

plt.scatter(residual_graph.y_pred, residual_studentized, alpha=0.1)
plt.xlabel('Predicted Values')
plt.ylabel('Residual Studentized')
plt.title('Figure 7: Graph of the (Residual Studentized x Predicted Values) Relationship')
plt.show()

'''
plt.hist(residual_studentized, bins=100)
plt.axvline(x=np.mean(residual_studentized), color='green')
plt.ylabel('Frequency')
plt.xlabel('Residual Studentized')
plt.show()
'''
sns.histplot(residual_studentized, bins=100, stat='density', kde=True)
plt.xlabel('Residual Studentized')
plt.title('Figure 8: Graph of the distribution curve of the Studentized Residual')
plt.show()

#Variance Inflation Factor Analysis

variables_name=['constant']+[x for x in flight_df_part.columns if x not in ['dow_date_scrapping', 'time_scrapping.hour', 'price_variation', 'dow_flight_departure', 'time_flight_departure.hour']]
variables=pd.DataFrame(X.astype(float))
VIF=pd.Series([variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])], index=variables.columns)
print("\n Variance Inflation Factor:\n", VIF)



#Breusch-Pagan Test
names = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
bp_test=sms.het_breuschpagan(residual_studentized, X)
print(names)
print(bp_test)

