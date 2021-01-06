#importing libraries
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix




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

X = traindf.drop(['PassengerId','Survived','Name','Ticket','Cabin','SibSp','Parch'], axis = 1) #dropping unnecessary columns
y = traindf['Survived']
x = testdf.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'], axis = 1)

#creating dummy variables
sex = pd.get_dummies(X['Sex'])
pclass = pd.get_dummies(X['Pclass'])
embarked = pd.get_dummies(X['Embarked'])
deck = pd.get_dummies(X['Deck'])
X = pd.concat([X, sex, pclass, embarked, deck], axis= 1)
X.drop(['Pclass','Sex','Embarked','Deck'], axis = 1, inplace = True)

sex = pd.get_dummies(x['Sex'])
pclass = pd.get_dummies(x['Pclass'])
embarked = pd.get_dummies(x['Embarked'])
deck = pd.get_dummies(x['Deck'])
x = pd.concat([x, sex, pclass, embarked, deck], axis= 1)
x.drop(['Pclass','Sex','Embarked','Deck'], axis = 1, inplace = True)
x['T'] = 0

#splitting the train data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 101)

#training the logistic regression model
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#predicting values of the test set
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))

#predicting values of a validating set and delivering an accuracy of 72.727%
predictions = logmodel.predict(x)