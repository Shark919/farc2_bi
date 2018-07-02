import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Binarizer, scale
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cross_validation as cv
from sklearn import tree
from sklearn import naive_bayes

printToCSV = False
train = ""
test = ""
X_train = ""
X_test = ""
y_train = ""
y_test = ""
selectedFeatures = ""
target = 'TARGET'

def main():
    initializeData()
    #dataUnderstanding()
    dataPreparation()
    #modeling()
    #evaluation()

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
    sns.countplot(train["TARGET"])
    plt.show()
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


def dataPreparation():
    """
    - spalten rauslöschen die identische werte haben
    - spalten rauslöschen mit geringer varianz
    - spalten rauslöschen die untereinander stark korrelieren
    - Zeilen rauslöschen mit fehlenden werten
    """
    global train
    global test

    removeConstantColumns(train, test)
    removeDuplicatedColumns(train, test)
    printToCSVWithFilename(train, 'train_cleanup.csv')
    printToCSVWithFilename(test, 'test_cleanup.csv')
    deleteColumnsWithHighCorrelation(train)
    deleteColumnsWithHighCorrelation(test)
    printToCSVWithFilename(train, 'train_remove_high_correlation.csv')
    printToCSVWithFilename(test, 'test_remove_high_correlation.csv')
    featureSelection()
    dataVisualization()
    removeRowsMissingValues()
    
    #### data cleansing is not necessary in our case, only numeric values
    #normalize()
    
    
def dataVisualization():
    ## Heatmap visualization of correlations
    sns.heatmap(train.corr())
    plt.show()
    for column in train:
        if len(train[column].unique()) < 10 and column != "TARGET":
            sns.countplot(train[column])
            plt.show()


def removeRowsMissingValues():
    print("Removing Rows with missing values...")
    for index, row in train.iterrows():
        for column in train:
            if row[column] == -999999 or row[column] == 999999 or row[column] == 9999999999 or row[column] == -9999999999:
                train.drop(index, inplace = True)
                break
    print("Rows with missing values removed.")    
    
def modeling():
    splitDataset()
    logisticRegression()
    #decisionTreeClassifier()

def evaluation():
    print("todo")
    # todo
    
def featureSelection():    
    # EITHER CHI OR DECISON
    chiSquared()
    #printToCSVWithFilename(train, 'train_f_sel_decision_tree.csv')
    #printToCSVWithFilename(test, 'test_f_sel_decision_tree.csv')
    #decisionTree()
    #decisionForest()

def removeConstantColumns(train, test):
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    
def removeDuplicatedColumns(train, test):
    remove = []
    cols = train.columns
    for i in range(len(cols)-1):
        v = train[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,train[cols[j]].values):
                remove.append(cols[j])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    
def deleteColumnsWithHighCorrelation(data):
    print('Delete Columns with high correlation...')
    corr_matrix = data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.85
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    for col in to_drop:
        del data[col]

def decisionTree():
    global test
    global train
    global selectedFeatures
        
    test = test.drop(["ID"],axis=1)

    X = train.drop(["TARGET","ID"],axis=1)
    y = train.TARGET.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1729)
    print(X_train.shape, X_test.shape, test.shape)

    clf = ExtraTreesClassifier(random_state=1729)
    selector = clf.fit(X_train, y_train)
    #fs = SelectFromModel(selector, prefit=True)
    
    feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)
    importantFeatures = feat_imp[:20]
    
    train = train[importantFeatures.index.tolist()+['TARGET']]
    test = test[importantFeatures.index.tolist()]
    
    #X_train = fs.transform(X_train)
    #X_test = fs.transform(X_test)
    #scaled_features = fs.transform(X)
    #selectedFeatures = pd.DataFrame(scaled_features, index=train.index, columns=train.columns)
    #selectedFeatures = pd.DataFrame(scaled_features, columns = X.columns)
    #train = fs.transform(train)
    #print(X_train.shape, X_test.shape, test.shape)

def chiSquared():
    global train
    global test
    data = train.iloc[:,:-1]
    y = train.TARGET
    binarizedData = Binarizer().fit_transform(scale(data))
    selectChi2 = SelectPercentile(chi2, percentile=3).fit(binarizedData, y)
    
    chi2_selected = selectChi2.get_support()
    chi2_selected_features = [ f for i,f in enumerate(data.columns) if chi2_selected[i]]
    print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
       chi2_selected_features))
    train = train[chi2_selected_features+['TARGET']]
    test = test[chi2_selected_features]
    
#def dataVisualization():
    # todo

def logisticRegression():
    ###CHECK ON TRAIN
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print('logistic regression score: ')
    print(score)

    #ynew = model.predict(X_test)
    #numpy.savetxt("logistic_regression_train.csv", ynew, delimiter=";")
   
    #printToCSVWithFilename(ynew, 'logistic_regression_train.csv')
    
    ###PREDICT ON TEST
    trainWithoutTarget = train.iloc[:,:-1]
    model = LogisticRegression()
    model.fit(trainWithoutTarget, train.TARGET)
    ynew = model.predict(trainWithoutTarget)
    # show the inputs and predicted outputs
    #for i in range(len(trainWithoutTarget)):
    #	print("X=%s, Predicted=%s" % (test.values[i], ynew[i]))
    printToCSVWithFilename(ynew, 'result.csv')
    
def decisionTreeClassifier():
    feature_names = train.columns.tolist()
    feature_names.remove('TARGET')
    
    X = train[feature_names]
    y = train[target]
    skf = cv.StratifiedKFold(y, n_folds=3, shuffle=True)
    score_metric = 'roc_auc'
    scores = {}
    
    def score_model(model):
        return cv.cross_val_score(model, X, y, cv=skf, scoring=score_metric)
        
    # time: 10s
    scores['tree'] = score_model(tree.DecisionTreeClassifier()) 
    scores['gaussian'] = score_model(naive_bayes.GaussianNB())
    
    model_scores = pd.DataFrame(scores).mean()
    model_scores.sort_values(ascending=False)
    model_scores.to_csv('model_scores.csv', index=False)
    print('Model scores\n{}'.format(model_scores))

def splitDataset():
    global X_train
    global X_test
    global y_train
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:-1], train.TARGET, test_size=0.20, random_state=1729)

def printToCSVWithFilename(data, filename):
    if printToCSV:
        data.to_csv(filename, sep=';', encoding='utf-8')

def histogram(data, x_label, y_label, title):
    _, ax = plt.subplots()
    ax.hist(data, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

main()
