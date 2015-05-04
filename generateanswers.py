import pandas as pd
import sys
import numpy as np
answers = pd.read_csv('answers.csv',header=0)
test = pd.read_csv('test.csv', header=0)

answers=answers.values.tolist()
answers_dict={}
for item in answers:
    name = item[2]
    if name is not np.NaN:
        name = name.replace('"','')
    answers_dict[name]=item[1]

test_answers=[]
nameid=[]
surival=[]
test = test.values.tolist()
for item in test:
    name=item[2]
    name = name.replace('"','')
    number=item[0]
    if name in answers_dict:
        survived=answers_dict[name]
    else:
        print number,name
        survived="lookup"
    nameid+=[int(number)]
    surival+=[survived]
    test_answers+=[[number,survived]]
nameid = pd.DataFrame(nameid)
answers = pd.DataFrame(surival, index=nameid)
answers.to_csv("answerstest.csv")



