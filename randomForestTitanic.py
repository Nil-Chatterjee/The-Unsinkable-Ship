#importing libraries
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score




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
y = traindf["Survived"]
X = traindf
X.drop(['Survived'], axis = 1, inplace = True)


sex = pd.get_dummies(testdf['Sex'])
pclass = pd.get_dummies(testdf['Pclass'])
embarked = pd.get_dummies(testdf['Embarked'])
deck = pd.get_dummies(testdf['Deck'])
testdf = pd.concat([testdf, sex, pclass, embarked, deck], axis= 1)
testdf.drop(['Pclass','Sex','Embarked','Deck'], axis = 1, inplace = True)
testdf['T'] = 0

#splitting the train data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

#defining the classifier model
rfc = RandomForestClassifier(random_state= 42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator= rfc, param_grid= param_grid, cv= 5)

#fitting the data to find the best parametres
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_

#defining the final classifier model with the best calculated parametres
rfcFinal = RandomForestClassifier(random_state = 42, max_features= 'auto', n_estimators = 500, max_depth= 7, criterion= 'entropy')

#training the model
rfcFinal.fit(X_train, y_train)

#predicting the result in the testing data set and a accuracy of 80.9701% is acchieved
pred= rfcFinal.predict(X_test)
print("Accuracy score: ",accuracy_score(y_test, pred))

#predicting the result of a validating data set an accuracy of 77.511% was achieved
predFinal = rfcFinal.predict(testdf)
submit= pd.DataFrame(PassengerId)
submit['Survived']= predFinal
submit.to_csv("submission3.csv", index= False)