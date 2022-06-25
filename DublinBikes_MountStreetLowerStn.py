import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import datetime
import warnings
warnings.filterwarnings("ignore")

plot = True
plt.rcParams['figure.constrained_layout.use'] = True
df = pd.read_csv("dublinbikes_20200101_20200401.csv", parse_dates=[1])
df = df.dropna()
df = df[df['NAME'] == "MOUNT STREET LOWER"]
# df = df[df['NAME'] == "ST. STEPHEN'S GREEN EAST"]
df = df[['TIME','AVAILABLE BIKES']]
start=pd.to_datetime("01-02-2020",format='%d-%m-%Y')
end=pd.to_datetime("01-04-2020",format='%d-%m-%Y')

# FEATURE ENGINEERING
# convert date/time to unix timestamp in sec
t_full=pd.array(pd.DatetimeIndex(df.iloc[:,0]).astype(np.int64))/1000000000
dt = t_full[1]-t_full[0]
print("data sampling interval is %d secs"%dt)
# extract data between start and end dates
t_start = pd.DatetimeIndex([start]).astype(np.int64)/1000000000
t_end = pd.DatetimeIndex([end]).astype(np.int64)/1000000000
t = np.extract([(t_full>=t_start[0]).astype(np.int64) & (t_full <= t_end[0]).astype(np.int64)], t_full)

t=(t-t[0])/60/60/24 # convert timestamp to days
y = np.extract([(t_full>=t_start[0]).astype(np.int64) & (t_full<=t_end[0]).astype(np.int64)], df.iloc[:,1])

q=12
# q=6
# q=2
lag=3; stride=1
w=math.floor(7*24*60*60/dt) # number of samples per week 2016
len = y.size-w-lag*w-q
XX=y[y.size-(lag*w)-len:y.size-(lag*w):stride]
for i in range(1,lag):
    X=y[y.size-((lag-i)*w)-len:y.size-((lag-i)*w):stride]
    XX=np.column_stack((XX,X))

d=math.floor(24*60*60/dt) # number of samples per day
for i in range(0,lag):
    X=y[y.size-((lag-i)*d)-len:y.size-((lag-i)*d):stride]
    XX=np.column_stack((XX,X))

sd=1
for i in range(0,lag):
    X=y[y.size-((lag-i-1)*sd)-q-len:y.size-((lag-i-1)*sd)-q:stride]
    XX=np.column_stack((XX,X))

yy=y[lag*w+w+q:lag*w+w+q+len:stride]
tt=t[lag*w+w+q:lag*w+w+q+len:stride]

train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)

# Ridge model without polynomial and hyperparameters
model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
print("Model Intercept: ",model.intercept_)
print("Coefficients of the model:")
print(model.coef_)
if plot:
    y_pred_rr = model.predict(XX)
    print('R2score of Ridge model: ',round(r2_score(yy, y_pred_rr),2))
    print('Mean squared error of Ridge model:',round(mean_squared_error(yy, y_pred_rr),2))
    print('Root Mean squared error of Ridge model:', round(np.sqrt(mean_squared_error(yy, y_pred_rr)), 2))
    print('Mean absolute error of Ridge model:', round(mean_absolute_error(yy, y_pred_rr),2))
    plt.scatter(t, y, color='black', marker='.');
    plt.scatter(tt, y_pred_rr, color='blue', marker='.', alpha=0.5)
    plt.xlabel("time (days)"); plt.ylabel("#bikes")
    plt.legend(["training data","predictions"],loc='upper right')
    day=math.floor(24*60*60/dt) # number of samples per day
    plt.xlim((5 * 7, 9 * 7))
    plt.show()

# Cross validations to identify polynomial degree and hyperparameter C values
q_range = [1,2,3,4]
mean_error_q = []
std_error_q = []
for pol in q_range:
    print("For polynomial: ",pol)
    Xpoly = PolynomialFeatures(pol).fit_transform(XX)
    mean_error_rr = []
    std_error_rr = []
    Ci_range = [0.0001,0.001,0.01,0.1,0.5,1]
    temp_rr = []
    for Ci in Ci_range:
        model = Ridge(alpha=1/(2*Ci))
        model.fit(Xpoly[train], yy[train])
        scores = cross_val_score(model, Xpoly[train], yy[train], cv=5, scoring='neg_mean_squared_error')
        mean_error_rr.append(scores.mean())
        std_error_rr.append(scores.std())

        print("C= %.4f, Neg Mean square error= % 0.2f(+ /− % 0.2f)"%(Ci, scores.mean(), scores.std()))
    plt.errorbar(Ci_range, mean_error_rr, yerr=std_error_rr)
    plt.xlabel("C")
    plt.ylabel("Mean square error")
    plt.title("Error Bar For q %d"%(pol))
    plt.xlim((0, 1.1))
    plt.show()
    print("-----------------------------------------------------------------------")
    mean_error_q.append(mean(mean_error_rr))
    std_error_q.append(mean(std_error_rr))
plt.errorbar(q_range, mean_error_q, yerr=std_error_q)
plt.xlabel("Q")
plt.ylabel("Mean square error")
plt.title("Error Bar For q ")
plt.xlim((0, 6))
plt.show()

# Ridge model with polynomial degree 3 and hyperparameter C=0.0001
C=0.0001
pol=3
Xpoly = PolynomialFeatures(pol).fit_transform(XX)
model = Ridge(alpha=1/(2*C)).fit(Xpoly[train], yy[train])
print("Model Intercept: ",model.intercept_)
print("Coefficients of the model:")
print(model.coef_)
if plot:
    y_pred = model.predict(Xpoly)
    print("For polynomial %d and C=%.4f"%(pol,C))
    print('R2score of Ridge model is ',round(r2_score(yy, y_pred), 2))
    print('Mean squared error of Ridge model:', round(mean_squared_error(yy, y_pred), 2))
    print('Root Mean squared error of Ridge model:', round(np.sqrt(mean_squared_error(yy, y_pred)), 2))
    print('Mean absolute error of Ridge model:', round(mean_absolute_error(yy, y_pred), 2))
    plt.scatter(t, y, color='black', marker='.');
    plt.scatter(tt, y_pred, color='blue', marker='.', alpha=0.5)
    plt.xlabel("time (days)"); plt.ylabel("#bikes")
    plt.legend(["training data","predictions"],loc='upper right')
    day=math.floor(24*60*60/dt) # number of samples per day
    plt.xlim((5 * 7, 9 * 7))
    plt.show()

# Cross Validation to get appropriate value of k for kNN
k_range = [3,5,7,9]
mean_error_knn = []
std_error_knn = []
for k in k_range:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(XX[train], yy[train])
    scores = cross_val_score(model, XX[train], yy[train], cv=5, scoring='neg_mean_squared_error')
    mean_error_knn.append(scores.mean())
    std_error_knn.append(scores.std())
    print("k= %.2f, Neg Mean square error= % 0.2f(+ /− % 0.2f)" % (k, scores.mean(), scores.std()))
plt.errorbar(k_range, mean_error_knn, yerr=std_error_knn)
plt.xlabel("k")
plt.ylabel("Mean square error")
plt.title("Error Bar ")
plt.xlim((2, 10))
plt.show()

# kNN model with k=3
k=3
model = KNeighborsRegressor(n_neighbors=k).fit(XX[train], yy[train])
if plot:
    y_pred_knn = model.predict(XX)
    print("For kNN with k ",k)
    print('R2score of kNN model ',round(r2_score(yy, y_pred_knn), 2))
    print('Mean squared error of kNN model:', round(mean_squared_error(yy, y_pred_knn), 2))
    print('Root Mean squared error of kNN model:', round(np.sqrt(mean_squared_error(yy, y_pred_knn)), 2))
    print('Mean absolute error of kNN model:', round(mean_absolute_error(yy, y_pred_knn), 2))
    plt.scatter(t, y, color='black', marker='.');
    plt.scatter(tt, y_pred_knn, color='blue', marker='.', alpha=0.5)
    plt.xlabel("time (days)"); plt.ylabel("#bikes")
    plt.legend(["training data","predictions"],loc='upper right')
    day=math.floor(24*60*60/dt) # number of samples per day
    plt.xlim((5 * 7, 9 * 7))
    plt.show()

# Baseline Model
def model_baseline(x):
	return x[-1]

y_pred_bm = list()
for x in XX:
	yhat = model_baseline(x)
	y_pred_bm.append(yhat)

print("For Baseline")
print('R2score of Baseline model ',round(r2_score(yy, y_pred_bm), 2))
print('Mean squared error of Baseline model:', round(mean_squared_error(yy, y_pred_bm), 2))
print('Root Mean squared error of Baseline model:', round(np.sqrt(mean_squared_error(yy, y_pred_bm)), 2))
print('Mean absolute error of Baseline model:', round(mean_absolute_error(yy, y_pred_bm), 2))
plt.scatter(t, y, color='black', marker='.');
plt.scatter(tt, y_pred_bm, color='blue', marker='.', alpha=0.5)
plt.xlabel("time (days)"); plt.ylabel("#bikes")
plt.legend(["training data","predictions"],loc='upper right')
day=math.floor(24*60*60/dt) # number of samples per day
plt.xlim((5 * 7, 9 * 7))
plt.show()