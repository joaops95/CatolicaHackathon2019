from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df_test = pd.read_csv('./test.csv')
df_train = pd.read_csv('./train.csv')


def normalizeData(df):
  passengerIdRow = df['passengerid']
  df.drop(['ticket', 'name', 'passengerid', 'cabin'], axis = 1, inplace=True)
  df['fare'].fillna(np.median(df['fare']),inplace=True)
  df['age'].fillna(np.mean(df['age']),inplace=True)
  df['embarked'].fillna((1),inplace=True)                  
  df['fare'] = df['fare']/(max(df['fare']))
  df['age'] = df['age']/(max(df['age']))
  for row in range(0, len(df)):
    if(df['sex'].iloc[row] == 'male'):
      df['sex'].iloc[row] = 0;
    else:
      df['sex'].iloc[row] = 1;  
    if(df['embarked'].iloc[row] == 'C'):
      df['embarked'].iloc[row] = 0;
    elif(df['embarked'].iloc[row] == 'Q'):
      df['embarked'].iloc[row] = 1;
    elif(df['embarked'].iloc[row] == 'S'):
      df['embarked'].iloc[row] = 2;
  df.dropna(inplace=True)
  return df, passengerIdRow.values.tolist()


k_range = range(1,30)
scores = {}
scores_list = []

df_train, train_passenger_row = normalizeData(df_train)

df_test, test_passenger_row  = normalizeData(df_test)
# print(len(df_test))
# for cname in df_train.columns:
#   print(df_train[cname].unique())
# print(df_train.head())

y = df_train['survived'].values
x = df_train.drop(['survived'], axis = 1).values


x_testfinal = df_test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=None)


for k in k_range:
  knn = KNeighborsClassifier(n_neighbors = k)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)
  scores[k] = metrics.accuracy_score(y_test, y_pred)
  scores_list.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores_list)
plt.xlabel('K value')
plt.ylabel('Accuracy')
newk = scores_list.index(max(scores_list))
print(newk)
knn = KNeighborsClassifier(n_neighbors = newk)
knn.fit(x_train, y_train)

y_pred_final = knn.predict(x_testfinal)


data = {'passengerid': test_passenger_row, 'survived': y_pred_final}

df_to_submit = pd.DataFrame(data) 
print(df_to_submit.head())
df_to_submit.to_csv(path + 'submition.csv', index = False)
print('done')