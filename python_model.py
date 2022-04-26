
#------------------------------------------------------------------------------------------------
# PROGRAM DESCRIPTION
# KAGGLE TITANIC SUBMISSION SCRIPT
# CURRENTLY USING RANDOM FORREST AND TWEAKING PARAMETERS
#------------------------------------------------------------------------------------------------



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# random forest model
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv("data/train.csv")
train_data.head()
test_data = pd.read_csv("data/test.csv")
test_data.head()


"""
# Find % of women that survived 
# the brackets are meant to be like that
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# Find % of women that survived 
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
"""









y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")