from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
import csv_io
import math
import scipy
import operator
def main():
    train=csv_io.read_data("Data/train.csv")
    train=train[0:10000]
    target=[x[0] for x in train]
    train=[x[1:] for x in train]
    realtest=csv_io.read_data("Data/test.csv")
    forest = RandomForestClassifier(n_estimators = 100)
    forest=forest.fit(train,target)
    predicted_probs=forest.predict_proba(realtest)
    fr=open('Result2.csv','w')
    fr.write("ImageId,Label"+"\n")
    count=1
    for y in predicted_probs:
        index, value = max(enumerate(y), key=operator.itemgetter(1))
        fr.write(str(count)+","+str(index)+"\n")
        count+=1
    fr.close()
if __name__=="__main__":
    main()
