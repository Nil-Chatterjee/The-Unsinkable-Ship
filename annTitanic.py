#importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score




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
X= traindf.drop(["Survived"], axis= 1).iloc[:,:].values
y = traindf.iloc[:, 0].values


sex = pd.get_dummies(testdf['Sex'])
pclass = pd.get_dummies(testdf['Pclass'])
embarked = pd.get_dummies(testdf['Embarked'])
deck = pd.get_dummies(testdf['Deck'])
testdf = pd.concat([testdf, sex, pclass, embarked, deck], axis= 1)
testdf.drop(['Pclass','Sex','Embarked','Deck'], axis = 1, inplace = True)
testdf['T'] = 0

#splitting the train data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#creating neural network with 2 hidden layers.
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN model and predicting
ann.fit(X_train, y_train, batch_size = 10, epochs = 200)
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

#calculating metrics with the predicted results for test set and an accuracy of 81.56% is achieved
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))