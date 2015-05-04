""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import tree
from sklearn import cross_validation


def prepareData(data):
    
    # Data cleanup
    

    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.

    # female = 0, Male = 1
    data['Gender'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    

    data['embark_c'] = data.Embarked.map(lambda x: 1 if x=="C" else 0)
    data['embark_q'] = data.Embarked.map(lambda x: 1 if x=="Q" else 0)
    data['embark_s'] = data.Embarked.map(lambda x: 1 if x=="S" else 0)
    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(data.Embarked[ data.Embarked.isnull() ]) > 0:
        data.Embarked[ data.Embarked.isnull() ] = data.Embarked.dropna().mode().values

    Ports = list(enumerate(np.unique(data['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    data.Embarked = data.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    

    # All the ages with no data -> make the median of all Ages
    median_age = data['Age'].dropna().median()
    if len(data.Age[ data.Age.isnull() ]) > 0:
        data.loc[ (data.Age.isnull()), 'Age'] = -100

    # All the missing Fares -> assume median of their respective class
    if len(data.Fare[ data.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            median_fare[f] = data[ data.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3):                                              # loop 0 to 2
            data.loc[ (data.Fare.isnull()) & (data.Pclass == f+1 ), 'Fare'] = median_fare[f]
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Embarked'], axis=1)

    return data

if __name__ == '__main__':
    # TRAIN DATA
    train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe
    # TEST DATA
    test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

    answers= pd.read_csv('answerstest.csv', header=0)
    answers_targets = answers['Survived'].values

    # Collect the test data's PassengerIds before dropping it
    ids = test_df['PassengerId'].values

    train_df = prepareData(train_df)
    test_df = prepareData(test_df)

    print train_df.columns

    # The data is now ready to go. So lets fit to the train, then predict to the test!
    # Convert back to a numpy array
    train_data = train_df.values
    test_data = test_df.values

    print 'Training...'
    forest = AdaBoostClassifier(n_estimators=250)
    #forest = RandomForestClassifier(n_estimators=250)
    #forest = tree.DecisionTreeClassifier()

    scores = cross_validation.cross_val_score(forest, train_data[0::,1::], train_data[0::,0], cv=5)
    print 'cross validate scores',scores


    forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
    
    print 'Predicting...'
    output = forest.predict(test_data).astype(int)
    print len(test_data)
    print len(answers_targets)
    score = forest.score(test_data,answers_targets)
    print "score",score


    predictions_file = open("myfirstforest.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'
    
