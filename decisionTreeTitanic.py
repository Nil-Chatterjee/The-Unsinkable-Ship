#importing libraries
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn import tree
from sklearn.model_selection import KFold




#function to return a particular substring from a string
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print (big_string)
    return np.nan

#data cleaning and data engineering by imputing values and/or by making other changes
def clean1(df):
    #setting silly values to nan
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    
    #Special case for cabins as nan may be signal
    df.Cabin = df.Cabin.fillna('Unknown')

    #Turning cabin number into Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
        
    #Creating new family_size column
    df['Family_Size']=df['SibSp']+df['Parch']
    
    return df

def clean2(train, test):

    for df in [train, test]:
        classmeans = df.pivot_table('Fare', columns='Pclass', aggfunc='mean')
        df.Fare = df[['Fare', 'Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1 )
        meanAge=np.mean(df.Age)
        df.Age=df.Age.fillna(meanAge)
        modeEmbarked = mode(df.Embarked)[0][0]
        df.Embarked = df.Embarked.fillna(modeEmbarked)
    
    return [train,test]

#reading the .csv files
trainpath = 'C:/Users/LuciferX10/.spyder-py3/train.csv'
testpath = 'C:/Users/LuciferX10/.spyder-py3/test.csv'
traindf = pd.read_csv(trainpath)
testdf = pd.read_csv(testpath)

traindf=clean1(traindf)
testdf=clean1(testdf)

traindf, testdf =clean2(traindf, testdf)

PassengerId = testdf['PassengerId']

#dropping unnecessary columns
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
traindf = traindf.drop(drop_elements, axis = 1)
testdf  = testdf.drop(drop_elements, axis = 1)

#creating dummy variables
sex = pd.get_dummies(traindf['Sex'])
pclass = pd.get_dummies(traindf['Pclass'])
embarked = pd.get_dummies(traindf['Embarked'])
deck = pd.get_dummies(traindf['Deck'])
traindf = pd.concat([traindf, sex, pclass, embarked, deck], axis= 1)
traindf.drop(['Pclass','Sex','Embarked','Deck'], axis = 1, inplace = True)

sex = pd.get_dummies(testdf['Sex'])
pclass = pd.get_dummies(testdf['Pclass'])
embarked = pd.get_dummies(testdf['Embarked'])
deck = pd.get_dummies(testdf['Deck'])
testdf = pd.concat([testdf, sex, pclass, embarked, deck], axis= 1)
testdf.drop(['Pclass','Sex','Embarked','Deck'], axis = 1, inplace = True)
testdf['T'] = 0

cv = KFold(n_splits=10)            # Desired number of Cross Validation folds
accuracies = list()
max_attributes = len(list(testdf))
depth_range = range(1, max_attributes + 1)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(traindf):
        f_train = traindf.loc[train_fold] # Extract train data with cv indices
        f_valid = traindf.loc[valid_fold] # Extract valid data with cv indices

        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 
                               y = f_train["Survived"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 
                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    print("Accuracy per fold: ", fold_accuracy, "\n")
    print("Average accuracy: ", avg)
    print("\n")
    
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))

y_train = traindf['Survived']
x_train = traindf.drop(['Survived'], axis=1).values 
x_test = testdf.values

decision_tree = tree.DecisionTreeClassifier(max_depth = 6)
decision_tree.fit(x_train, y_train)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print(acc_decision_tree) #an accuracy score of 86.98% is achieved

#predicting values of a validating data set and an accuracy of 74.162% is achieved
y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)