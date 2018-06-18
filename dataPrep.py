# Based on https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np
#import datetime
from sklearn import datasets, linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import ExtraTreesClassifier
#from matplotlib import pyplot as plt

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

train = ""
test = ""

def main():
    initializeData()
    dataUnderstanding()
    dataPreperation()
    #modeling()
    evaluation()

def initializeData():
    print('Reading data...')
    global train
    global test
    train = pd.read_csv("./Data/train.csv")
    test = pd.read_csv("./Data/test.csv")
    print('Finished reading data.')
 
def dataUnderstanding():
    print(train.info(verbose=True, null_counts=True))
    print("")
    print("Zielvariablenbeschreibung...:")
    print(train["TARGET"].describe())
    print(train["TARGET"].unique())
    anzahl = train["TARGET"].value_counts()
    print("Anzahl Satisfied: ", anzahl[0])
    print("Anzahl Unsatisfied: ", anzahl[1])
    print("Anzahl in Prozent: ", (anzahl[0]/(anzahl[0]+anzahl[1]))*100,"%.")
    # look if columns contain distinct variables
    dist_counter = 0
    for column in train:
        if len(train[column].unique()) < 2: # später für data preparation
        #print(column)
            dist_counter = dist_counter + 1
    
    print(dist_counter , " Spalten beinhalten keine unterschiedlichen Werte!")
    """ 34 Spalten beinhalten keine unterschiedlichen Werte"""
    # look if columns contain string format
    string_counter = 0
    for column in train:
        for i in train[column]:
            if type(i) is str:
                print(column)
                string_counter = string_counter + 1
    print(string_counter, " Spalten beinhalten String Werte!")
    
    nan_counter = train.isnull().values.sum()
    print(nan_counter," null-Werte sind insgesamt im Datensatz enthalten")
    for column in train:
        print(column ," ", " Max Wert: ",train[column].max()," Min Werte: ", train[column].min())
    """
    for column in train
        for i in train[column]:
            if i == -999999 or i == 999999 or i == 9999999999 or i == -9999999999:
                print("spaltennamen: ", column, "Wert: ", i, "andere werte", train[column].unique())
            else:
                pass
    """
    dataPreparation()
    #removeFeaturesWithLowVariance()
    #correlationToTarget()  # pointless??
    #deleteColumnsWithHighCorrelation()
    
def dataPreparation():
    """
    - spalten rauslöschen die identische werte haben
    - spalten rauslöschen mit geringer varianz
    - spalten rauslöschen die untereinander stark korrelieren
    - Zeilen rauslöschen mit fehlenden werten
    """
    """spalten rauslöschen die keine verschiedenen Werte beinhalten"""
    for column in train:
        if len(train[column].unique()) < 2: # später für data preparation
            del train[column]
        else:
            pass
    #featureSelection()
    #### data cleansing is not necessary in our case, only numeric values
    #normalize()
    #dataVisualization()

#def dataVisualization():
    # todo

#def modeling():
    # todo

def evaluation():
    trainAndTest()
    # todo

def removeFeaturesWithLowVariance():
    global train
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    train = pd.DataFrame(sel.fit_transform(train.values))
    file_name = 'remLowVariance.csv' #-{date:%Y-%m-%d-%H:%M:%S}    .format( date=datetime.datetime.now() )
    train.to_csv(file_name, sep=';', encoding='utf-8')

def correlationToTarget():
    print("Show Pearson's correlation:")
    c = train[train.columns[1:]].corr()['TARGET'][:-1]
    for i,r in c.items():
        print(i,r)
    #print("Show Spearman's rho correlation:")
    #print(train.corr('spearman'))
    
    #print("Show Kendal's tau correlation:")
    #print(train.corr('kendall'))
        
def deleteColumnsWithHighCorrelation():
    global train
    print('Delete Columns with high correlation...')
    corr_matrix = train.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.85
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    #print(to_drop)
    for col in to_drop:
        del train[col]
    file_name = 'corOut.csv' #-{date:%Y-%m-%d-%H:%M:%S}    .format( date=datetime.datetime.now() )
    train.to_csv(file_name, sep=';', encoding='utf-8')

def featureSelection():
    #chiSquared()
    decisionTree()

def decisionTree():
    global test
    # split data into train and test
    test_id = test.ID
    test = test.drop(["ID"],axis=1)

    X = train.drop(["TARGET","ID"],axis=1)
    y = train.TARGET.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)
    print(X_train.shape, X_test.shape, test.shape)

    ## # Feature selection
    clf = ExtraTreesClassifier(random_state=1729)
    selector = clf.fit(X_train, y_train)
    print(selector)

def chiSquared():
    ########
    # note: chi squared feature selection only works with positive values
    # we also have negative values
    # i dont know if there are feature selection methods that also work here
    # either we dont do this step OR normalize the data to postive values only
    ########
    
    values = train.values
    array = normalize(values, axis=0, norm='max')
    
    #Split the data into input and target
    X = array[:,:array.shape[1]-1]
    Y = array[:,[array.shape[1]-1]]
    test = SelectKBest(score_func=chi2, k=20)    
    fit = test.fit(X, Y) 
    #Summarize scores numpy.set_printoptions(precision=3) print(fit.scores_)
    
    #Apply the transformation on to dataset 
    features = fit.transform(X) 
    print(features[0:20,:])


#def normalize():
    # see feature selection, what to do with negative values?
    

def trainAndTest():
    # Let's see what's in the trainings data - Jupyter notebooks print the result of the last thing you do
    print('training with data')
    #print(train.head())
    #print('trained')
    
    #df = pd.DataFrame(train.TARGET.value_counts())
    #df['Percentage'] = 100*df['TARGET']/train.shape[0]
    #print(df)
    
    ####
    # todo: split train in half, fit first part on second and then predict on
    # test dataset
    ####
    
    #X_train, X_test, y_train, y_test = train_test_split(train, train.TARGET, test_size=0.2)
    # fit a model
    #lm = linear_model.LinearRegression()
    
    #model = lm.fit(X_train, X_test)
    #predictions = lm.predict(test)
    #print(predictions[0:5])
    #print (model.score(X_test, test))

main()
