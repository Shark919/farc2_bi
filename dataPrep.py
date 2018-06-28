# Based on https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import Binarizer, scale

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
    dataPreparation()
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
            dist_counter = dist_counter + 1
    print(dist_counter , " Spalten beinhalten keine unterschiedlichen Werte!")
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

    #dataPreparation()
    #removeFeaturesWithLowVariance()
    #correlationToTarget()  # pointless??
    #deleteColumnsWithHighCorrelation()

def removeConstantColumns():
    print("Removing Constant Columns...")
    global train
    global test
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    print("Constant Columns Removed.")

def removeDuplicatedColumns():
    print("Removing Column Duplicates...")
    global train
    global test
    remove = []
    cols = train.columns
    for i in range(len(cols)-1):
        v = train[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,train[cols[j]].values):
                remove.append(cols[j])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    print("Column Duplicates removed.")

def dataPreparation():
    #featureSelection()
    """spalten rauslöschen die identische werte haben"""
    removeConstantColumns()
    """redundante Spalten löschen"""
    removeDuplicatedColumns()
    """zeilen mit fehlenden Werten löschen"""
    removeRowsMissingValues()
    """neue csv für train datensatz"""
    printToCSVWithFilename(train, 'train_cleanup.csv')
    """neue csv für test datensatz"""
    printToCSVWithFilename(test, 'test_cleanup.csv')
    #removeFeaturesWithLowVariance()
    """ lösche eine der spalten wenn starke korrelation untereinander gegeben"""
    deleteColumnsWithHighCorrelation()
    featureSelection()
    #### data cleansing is not necessary in our case, only numeric values
    #normalize()
    #dataVisualization()

#def dataVisualization():
    # todo

#def modeling():
    # todo
def removeRowsMissingValues():
    print("Removing Rows with missing values...")
    for index, row in train.iterrows():
        for column in train:
            if row[column] == -999999 or row[column] == 999999 or row[column] == 9999999999 or row[column] == -9999999999:
                train.drop(index, inplace = True)
                break
    print("Rows with missing values removed.")
def evaluation():
    trainAndTest()
    # todo

def removeFeaturesWithLowVariance():
    global train
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    train = pd.DataFrame(sel.fit_transform(train.values))
    printToCSVWithFilename(train, 'train_remove_low_variance.csv')
        
def deleteColumnsWithHighCorrelation():
    global train
    print('Delete Columns with high correlation...')
    corr_matrix = train.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.85
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    for col in to_drop:
        del train[col]
    printToCSVWithFilename(train, 'train_remove_high_correlation.csv')

def featureSelection():
    # EITHER CHI OR DECISON
    chiSquared()
    #decisionTree()

def decisionTree():
    global test
    global train
    test = test.drop(["ID"],axis=1)

    X = train.drop(["TARGET","ID"],axis=1)
    y = train.TARGET.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)
    print(X_train.shape, X_test.shape, test.shape)

    clf = ExtraTreesClassifier(random_state=1729)
    selector = clf.fit(X_train, y_train)
    fs = SelectFromModel(selector, prefit=True)
    
    X_train = fs.transform(X_train)
    X_test = fs.transform(X_test)
    test = fs.transform(test)
    train = fs.transform(train)
    print(X_train.shape, X_test.shape, test.shape)
    printToCSVWithFilename(train, 'train_f_sel_decision_tree.csv')


def chiSquared():
    global train
    data = train.iloc[:,:-1]
    y = train.TARGET
    binarizedData = Binarizer().fit_transform(scale(data))
    selectChi2 = SelectPercentile(chi2, percentile=3).fit(binarizedData, y)
    
    chi2_selected = selectChi2.get_support()
    chi2_selected_features = [ f for i,f in enumerate(data.columns) if chi2_selected[i]]
    print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
       chi2_selected_features))
    train = train[chi2_selected_features+['TARGET']]
    printToCSVWithFilename(train, 'train_f_sel_chi_squared.csv')

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

def printToCSVWithFilename(data, filename):
    data.to_csv(filename, sep=';', encoding='utf-8')

main()
