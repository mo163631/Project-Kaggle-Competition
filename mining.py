import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('train.csv')

data.isna().sum()

data.drop('id', axis=1, inplace=True)

X = data.drop("Response", axis=1)
y = data["Response"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.0000333, random_state=42)

oversampler = RandomOverSampler(random_state=42)
x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train, y_train)

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('clf', XGBClassifier(random_state=1, learning_rate=0.02))  
])

pipeline.fit(x_train_resampled, y_train_resampled)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

model.fit(X, y)

test = pd.read_csv('test.csv')
t_id = test['id']
test.drop('id', axis=1, inplace=True)

y_predict = model.predict(test)

submission = pd.DataFrame({
    'id': t_id,
    'Response': y_predict
})

submission.to_csv("Data_Mining.csv", index=False)
