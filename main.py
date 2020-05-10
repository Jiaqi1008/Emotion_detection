import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm,tree,metrics
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


if __name__ =="__main__":
    df=pd.read_csv("feature.csv")
    feature_X=df.iloc[:,0:-3]
    feature_V=df.iloc[:,-3]
    feature_A=df.iloc[:,-2]
    feature_D=df.iloc[:,-1]
    X_train,X_test,Y_train,Y_test=train_test_split(feature_X,feature_V,test_size=0.3,random_state=7)
    # NN
    # model=MLPClassifier(activation='logistic',hidden_layer_sizes=(100,3),random_state=7)
    # SVM
    # model=svm.SVC(C=1,random_state=7)
    # DT
    model=tree.DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_leaf=2,min_samples_split=5,random_state=7)
    # GBDT
    # model=GBDT(learning_rate=0.1,max_depth=9,min_samples_leaf=60,min_samples_split=10,n_estimators=31,random_state=7)
    # ada
    # model=ada(LogisticRegression(penalty='l2',C=0.55,max_iter=1000),learning_rate=0.3,n_estimators=4,random_state=7)
    # LR 
    # model=LogisticRegression(penalty='l2',C=0.55,max_iter=1000)
    model.fit(X_train, Y_train)
    prediction=model.predict(X_test)
    print("Accuracy : %.4g" % metrics.accuracy_score(Y_test, prediction))