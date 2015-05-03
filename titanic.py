import csv 
import numpy as np
#actually dont need to scale for randomforest
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

csv_file_object = csv.reader(open('train.csv', 'rb')) 
header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data)            # Then convert from a list to an array
                                 # Be aware that each item is currently
                                 # a string in this format

target = data[0::,1]
features=data[0::,2::]
features = features.tolist()
trainsize=len(features)
finalfeatures=np.zeros((trainsize,8))
#male = 0
#female = 1
for x in range(len(features)):
    if features[x][2]=='male':
        features[x][2]=0
    else:
        features[x][2]=1
    
#miss /ms
#mrs
#mr
    titlevector=[0,0,0]
    name = features[x][1].lower()
    if 'miss.' in name or 'ms.' in name:
        titlevector[0]=1
    if 'mrs.' in name:
        titlevector[1]=1
    if 'mr.' in name:
        titlevector[2]=1

    features[x]=features[x][:-4]
    

    del features[x][1]
    features[x]+=titlevector
    #print features[x]
    finalfeatures[x]=np.array(features[x])

#normalize data


#train





