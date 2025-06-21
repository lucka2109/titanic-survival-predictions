# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Import the necessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 

import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
# See a sample of the dataset
train_data.head()

test_data = pd.read_csv(("/kaggle/input/titanic/test.csv"))
test_data.head()

# Get a list of the features within the training dataset 
print(train_data.columns)

# See a summary of the training dataset
train_data.describe(include= "all")

# Draw a barplot of survival by sex 
survival_rates = train_data.groupby('Sex')['Survived'].mean() * 100
plt.bar(survival_rates.index, survival_rates.values, color=['crimson', 'steelblue'])
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survived')
plt.yticks(range(0, 101, 10)) 
plt.show()

# Print percentages of females vs males who survived
print("Percentage of females who survived:", survival_rates['female'])
print("Percentage of males who survived:", survival_rates['male'])

# Draw a barplot of survival by passenger class
pclass_survival = train_data.groupby('Pclass')['Survived'].mean() * 100
plt.bar(pclass_survival.index, pclass_survival.values, color=['tan', 'cornflowerblue', 'teal'])
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.xticks([1, 2, 3])
plt.show()

# Print percentage of people by pclass that survived
print("Percentage of Pclass = 1 who survived:", pclass_survival[1])
print("Percentage of Pclass = 2 who survived:", pclass_survival[2])
print("Percentage of Pclass = 3 who survived:", pclass_survival[3])

# Draw a barplot of survival by SibSp 
sibsp_survival = train_data.groupby('SibSp')['Survived'].mean() * 100
colors = plt.cm.tab10(np.linspace(0, 1, len(sibsp_survival)))
plt.bar(sibsp_survival.index, sibsp_survival.values, color=colors)
plt.title('Survival Rate by SibSp')
plt.xlabel('SibSp')
plt.ylabel('Survived')
plt.xticks(sibsp_survival.index)
plt.show()

# Draw a barplot of survival by Parch
parch_survival = train_data.groupby('Parch')['Survived'].mean() * 100
colors = plt.cm.tab10(np.linspace(0, 1, len(parch_survival)))
plt.bar(parch_survival.index, parch_survival.values, color=colors)
plt.title('Survival Rate by Parch')
plt.xlabel('Parch')
plt.ylabel('Survived')
plt.ylim(0, 100)
plt.show()

# Fill missing ages with -0.5 (mark 'Unknown')
train_data["Age"] = train_data["Age"].fillna(-0.5)
test_data["Age"] = test_data["Age"].fillna(-0.5)

# Define bins and labels
bins = [-1, 0, 5, 12, 19, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Adult', 'Senior']

# Create AgeGroup categorical feature
train_data['AgeGroup'] = pd.cut(train_data["Age"], bins=bins, labels=labels)
test_data['AgeGroup'] = pd.cut(test_data["Age"], bins=bins, labels=labels)

# Calculate survival rate by AgeGroup 
age_survival = train_data.groupby('AgeGroup')['Survived'].mean() * 100

# Generate colors automatically using a colormap
colors = plt.cm.tab10(np.linspace(0, 1, len(age_survival)))

# Draw a barplot of survival by age
plt.bar(age_survival.index.astype(str), age_survival.values, color=colors)

plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survived')
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))
plt.show()

test_data.describe(include="all")

# Drop the cabin feature from both datasets
train_data = train_data.drop(['Cabin'], axis=1)
test_data = test_data.drop(['Cabin'], axis=1)

# Drop the ticket feature from both datasets 
train_data = train_data.drop(['Ticket'], axis=1)
test_data = test_data.drop(['Ticket'], axis=1)

print("Number of people embarking in Southampton (S):")
southampton = train_data[train_data["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg(C):")
cherbourg = train_data[train_data["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown(Q):")
queenstown = train_data[train_data["Embarked"] == "Q"].shape[0]
print(queenstown)

# Replacing the missing values in the Embarked feature with S
train_data = train_data.fillna({"Embarked": "S"})
test_data = test_data.fillna({"Embarked": "S"})

combine = [train_data, test_data] 

for dataset in combine: 
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])

# Group rare/unusual titles together
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Convert each title into a numeric value for modeling
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data.head()

# Fill missing age with mode age group for each title
mr_age = train_data[train_data["Title"] == 1]["AgeGroup"].mode() #Adult
miss_age = train_data[train_data["Title"] == 2]["AgeGroup"].mode() #Teenager
mrs_age = train_data[train_data["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train_data[train_data["Title"] == 4]["AgeGroup"].mode() #Child
royal_age = train_data[train_data["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train_data[train_data["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Adult", 2: "Teenager", 3: "Adult", 4: "Child", 5: "Adult", 6: "Adult"}

for x in range(len(train_data["AgeGroup"])):
    if train_data["AgeGroup"][x] == "Unknown":
        title = train_data["Title"][x]
        if title in age_title_mapping:
            train_data["AgeGroup"][x] = age_title_mapping[title]
        else:
            # Assign a default value for unknown titles
            train_data["AgeGroup"][x] = "Adult"
        
for x in range(len(test_data["AgeGroup"])):
    if test_data["AgeGroup"][x] == "Unknown":
        title = test_data["Title"][x]
        if title in age_title_mapping:
            test_data["AgeGroup"][x] = age_title_mapping[title]
        else:
            test_data["AgeGroup"][x] = "Adult"

# Map each age value to a numerical value
age_mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Adult': 4, 'Senior': 5}
train_data['AgeGroup'] = train_data['AgeGroup'].map(age_mapping)
test_data['AgeGroup'] = test_data['AgeGroup'].map(age_mapping)

train_data.head()

# Drop the age feature
train_data = train_data.drop(['Age'], axis = 1)
test_data = test_data.drop(['Age'], axis = 1)

# Drop the name feature
train_data = train_data.drop(['Name'], axis = 1)
test_data = test_data.drop(['Name'], axis = 1)

# Map each sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)

train_data.head()

# Map each embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)

train_data.head()

# Fill in missing fare value in test set with average fare of matching Pclass
for x in range(len(test_data["Fare"])):
    if pd.isnull(test_data["Fare"][x]):
        pclass = test_data["Pclass"][x] #Pclass = 3
        test_data["Fare"][x] = round(train_data[train_data["Pclass"] == pclass]["Fare"].mean(), 4)
        
# Divide fare into 4 equal-sized bands and label them 
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4, labels = [1, 2, 3, 4])
test_data['FareBand'] = pd.qcut(test_data['Fare'], 4, labels = [1, 2, 3, 4])

# Drop fare values
train_data = train_data.drop(['Fare'], axis = 1)
test_data = test_data.drop(['Fare'], axis = 1)

# Check train data
train_data.head()

# Check test_data
test_data.head()

from sklearn.model_selection import train_test_split

predictors = train_data.drop(['Survived', 'PassengerId'], axis=1)
target = train_data["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

# Train and evaluate a Gaussian Naive Bayes classifier on the data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)

# Train and evaluate a Logistic Regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)

# Train and evaluate a Random Forest model
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

# Train and evalute k-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)

# Train and evaluate a Stochastic Gradient Descent Classifier
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)

# Train and evaluate a Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

model_scores = [
    ('Naive Bayes', acc_gaussian), 
    ('Logistic Regression', acc_logreg),
    ('Random Forest', acc_randomforest),
    ('KNN', acc_knn),
    ('Stochastic Graident Descent Classifier', acc_sgd),
    ('Gradient Boosting Classifier', acc_gbk)
]

models_df = pd.DataFrame(model_scores, columns=['Model', 'Score']).sort_values(by='Score', ascending=False)

models_df

from IPython import get_ipython

code_cells = []

ip = get_ipython()
for cell in ip.history_manager.get_range():
    # cell is a tuple (session, line, input)
    code_cells.append(cell[2])

# Save all code cells into a .py file
with open("/kaggle/working/converted_script.py", "w") as f:
    for cell_code in code_cells:
        f.write(cell_code + "\n\n")

print("✅ All code saved to converted_script.py")

