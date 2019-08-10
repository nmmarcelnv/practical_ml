
## Practical Machine Learning Assignment


```python
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Summary of Prediction

**Note: I used Python, not R**

In this project, I used a Decision Tree classifier into a one-vs-one multiclass classifier. The reason I choosed a Tree-based model is because of the large number of feature in the data. Using a linear model sush as Logistic Regression or Support Vector Machine would have required one-hot encoing which would have increased the dimension even further. We do not have that problem with Tree-Based models, as label encoding works just fine. 

I started simple with a Decision Tree and it turned out to be the best model. I found that random forest overfitted the train set even with a small number of Trees in the forest. Also, I had to drop many features which had more than 50\% missing values, which reduced the dimention from 160 to 60. 

I used 10-fold cross validation to estimate the out-of-sample accuracy on the test set. 

### Data Import and Exploration


```python
df = pd.read_csv('pml-training.csv', low_memory=False)
```


```python
print(df.shape)
df[df.columns[0:10]].head()
```

    (19622, 160)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>user_name</th>
      <th>raw_timestamp_part_1</th>
      <th>raw_timestamp_part_2</th>
      <th>cvtd_timestamp</th>
      <th>new_window</th>
      <th>num_window</th>
      <th>roll_belt</th>
      <th>pitch_belt</th>
      <th>yaw_belt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>carlitos</td>
      <td>1323084231</td>
      <td>788290</td>
      <td>05/12/2011 11:23</td>
      <td>no</td>
      <td>11</td>
      <td>1.41</td>
      <td>8.07</td>
      <td>-94.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>carlitos</td>
      <td>1323084231</td>
      <td>808298</td>
      <td>05/12/2011 11:23</td>
      <td>no</td>
      <td>11</td>
      <td>1.41</td>
      <td>8.07</td>
      <td>-94.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>carlitos</td>
      <td>1323084231</td>
      <td>820366</td>
      <td>05/12/2011 11:23</td>
      <td>no</td>
      <td>11</td>
      <td>1.42</td>
      <td>8.07</td>
      <td>-94.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>carlitos</td>
      <td>1323084232</td>
      <td>120339</td>
      <td>05/12/2011 11:23</td>
      <td>no</td>
      <td>12</td>
      <td>1.48</td>
      <td>8.05</td>
      <td>-94.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>carlitos</td>
      <td>1323084232</td>
      <td>196328</td>
      <td>05/12/2011 11:23</td>
      <td>no</td>
      <td>12</td>
      <td>1.48</td>
      <td>8.07</td>
      <td>-94.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = df['classe']
df_clean = df.drop(['Unnamed: 0', 'classe'], axis=1)
```


```python
#find columns with large number of missing values
missing = []
for column in df_clean.columns:
    if df_clean[column].isna().sum()/df_clean.shape[0] > 0.5:
        missing.append(column)
df_clean = df_clean.drop(missing, axis=1)

objects = df_clean.select_dtypes(include=['object'])
numerics = df_clean.select_dtypes(include=['int', 'float64', 'int64'])
print(len(objects.columns) + len(numerics.columns) == len(df_clean.columns))
print(objects.shape, numerics.shape)
```

    True
    (19622, 3) (19622, 55)



```python
d = defaultdict(LabelEncoder)
categoric = objects.apply(lambda x: d[x.name].fit_transform(x))
```


```python
X_train = pd.concat([categoric, numerics], axis=1, sort=False)
y_train = LabelEncoder().fit_transform(y)
```


```python
predictor = OneVsOneClassifier(DecisionTreeClassifier(random_state = 1, max_depth = 5))
y_pred = predictor.fit(X_train, y_train).predict(X_train)
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
print(classification_report(y_train, y_pred, target_names=target_names))
```

                  precision    recall  f1-score   support
    
         class 0       0.93      0.89      0.91      5580
         class 1       0.90      0.78      0.83      3797
         class 2       0.86      0.92      0.89      3422
         class 3       0.78      0.90      0.83      3216
         class 4       0.93      0.94      0.93      3607
    
       micro avg       0.88      0.88      0.88     19622
       macro avg       0.88      0.88      0.88     19622
    weighted avg       0.89      0.88      0.88     19622
    


### Esimating Out-of-Sample Error with Cross-Validation Score  

Given the above results, I expect out-of-sample error to be significant. The above model will probably not generalize well to new data because the variance appears to be high. Let's use cross-validation to estimate out of sample error. 


```python
scores = cross_val_score(predictor, X_train, y_train, cv = 10)
print("Mean Cross-Validation Score: ", np.mean(scores))
```

    Mean Cross-Validation Score:  0.6595557261538972


### Test Set Performance


```python
df_test = pd.read_csv('pml-testing.csv', low_memory=False).drop(['Unnamed: 0'], axis=1)
print(df_test.shape)
df_test = df_test.drop(missing, axis=1)
print(df_test.shape)
objects_df = df_test.select_dtypes(include=['object'])
numerics_df = df_test.select_dtypes(include=['int', 'float64', 'int64'])
print(objects_df.shape, numerics_df.shape)
print(len(objects_df.columns) + len(numerics_df.columns) == len(df_test.columns))

categoric_df = objects_df.apply(lambda x: d[x.name].transform(x))
X_test = pd.concat([categoric_df, numerics_df], axis=1, sort=False)[X_train.columns]

y_test = predictor.predict(X_test)
```

    (20, 159)
    (20, 59)
    (20, 3) (20, 56)
    True



```python
y_test
```




    array([1, 0, 2, 0, 0, 4, 3, 3, 0, 0, 1, 2, 1, 0, 4, 4, 0, 3, 1, 1])




```python
y.unique()
```




    array(['A', 'B', 'C', 'D', 'E'], dtype=object)




```python
dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
results = pd.DataFrame({'prediction': y_test, 'classe': [dic[i] for i in y_test]})
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>classe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>E</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>C</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>E</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
      <td>E</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>B</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>


