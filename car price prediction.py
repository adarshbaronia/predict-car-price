import pandas as pd
df=pd.read_csv('Car Data.csv')

df['current_year'] = 2021
df['no_of_years'] = df['current_year']-df['Year']

df = df.drop(['Year','Car_Name','current_year'],axis=1)

final = pd.get_dummies(df, drop_first = True)

corr=final.corr()

import seaborn as sns
#sns.pairplot(final)

#sns.heatmap(corr,annot = True)

X = final.iloc[:,1:]
y = final.iloc[:,0]

# from sklearn.ensemble import ExtraTreesRegressor
# import matplotlib.pyplot as plt

# et = ExtraTreesRegressor()
# et.fit(X,y)

# print(et.feature_importances_)

# fi = pd.Series(et.feature_importances_, index = X.columns) 

# fi.nlargest(5).plot(kind = 'barh')
# plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.3, random_state = 0)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['sqrt','auto']
max_depth = [int(x) for x in np.linspace(start = 2, stop=30, num = 7)]
min_samples_split = [2,10,15,20,100]
min_samples_leaf =[1,2,5,10] 

random_grid = {'n_estimators':n_estimators,'max_features':max_features,
               'max_depth':max_depth, 'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}

rc = RandomizedSearchCV( estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error',n_iter = 10, cv=5, verbose = 2,random_state = 0, n_jobs=-1 )
rc.fit(xtrain, ytrain)

#print(rc.best_params_)
predict = rc.predict(xtest)
sns.displot(ytest-predict)

from sklearn.metrics import mean_absolute_error,mean_squared_error

print('MAE', mean_absolute_error(ytest, predict))

print('MSE', mean_squared_error(ytest, predict))

print('RMSE', np.sqrt(mean_squared_error(ytest, predict)))

import pickle
file = open('RandonForectRegressor.pkl','wb')
pickle.dump(rc,file)


