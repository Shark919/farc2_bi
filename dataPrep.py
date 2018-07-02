from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, scale
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn import cross_validation as cv
from sklearn import tree
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier as RF

featureSelectionType = "decision" #chi or decision
printToCSV = False

train = ""
test = ""
X_train = ""
X_test = ""
y_train = ""
y_test = ""
predictedValues = []

def main():
    print("CSV Mode activated: "+str(printToCSV))
    print("Feature Selection Type: "+str(featureSelectionType))
    initializeData()
    #dataUnderstanding()
    dataPreparation()
    modeling()
    evaluation()

def initializeData():
    print('STEP 0: Reading data')
    global train
    global test
    train = pd.read_csv("./Data/train.csv")
    test = pd.read_csv("./Data/test.csv")
 
def dataUnderstanding():
    print('STEP 1: Data Understanding')
    f = open('00_data_understanding.info', 'w+')
    train.info(verbose=True, null_counts=True, buf=f)
    f.writelines('-----Zielvariablenbeschreibung------\n')
    f.writelines(str(train["TARGET"].describe()))
    f.writelines(str(train["TARGET"].unique()) + '\n')
    anzahl = train["TARGET"].value_counts()
    f.writelines("Anzahl Satisfied: " + str(anzahl[0])+ '\n')
    f.writelines("Anzahl Unsatisfied: " + str(anzahl[1])+ '\n')
    f.writelines("Anzahl in Prozent: " + str((anzahl[0]/(anzahl[0]+anzahl[1]))*100) + "%.\n")
    f.writelines(str(countDistinct()) + " Spalten beinhalten keine unterschiedlichen Werte!\n")
    f.writelines(str(countStrings()) + " Spalten beinhalten String Werte!\n")
    f.writelines(str(train.isnull().values.sum()) + " null-Werte sind insgesamt im Datensatz enthalten!\n")
    for column in train:
        f.writelines(str(column) + "  Max Wert: " + str(train[column].max()) + "   Min Werte:  " + str(train[column].min())+ '\n')
    f.close()

    sns.countplot(train["TARGET"])
    plt.savefig('./Generated_Visualization/targetBarchart.png')
    print('Written data understanding file.')
    print('Generated barchart.')

def dataPreparation():
    print("STEP 2 :Data preparation")
    global train
    removeConstantColumns(train)
    removeDuplicatedColumns(train)
    printToCSVWithFilename(train, '01_train_cleanup.csv')
    deleteColumnsWithHighCorrelation(train)
    printToCSVWithFilename(train, '02_train_remove_high_correlation.csv')
    splitDataset()
    featureSelection()
    printToCSVWithFilename(train, '03_train_f_sel_decision_tree.csv')
    #dataVisualization()
    #removeRowsMissingValues()
    printToCSVWithFilename(train, '04_train_after_datapreperation.csv')
    #### data cleansing is not necessary in our case, only numeric values
    #### normalizing is not necessarry
    
def modeling():
    print('STEP 3: Modeling')
    global predictedValues
    feature_names = train.columns.tolist()
    feature_names.remove('TARGET')
    
    X = train[feature_names]
    y = train['TARGET']
    skf = cv.StratifiedKFold(y, n_folds=3, shuffle=True)
    score_metric = 'roc_auc'
    scores = {}
    
    def score_model(model, title):
        predicted = cv.cross_val_predict(model, X, y, cv=skf)
        predictedValues.append([title, predicted]);
        return cv.cross_val_score(model, X, y, cv=skf, scoring=score_metric)
    
    print('Logistic Regression...')
    scores['logistic'] = score_model(LogisticRegression(), 'logistic_regression')
    print('Decision Tree Classifier...')
    scores['tree'] = score_model(tree.DecisionTreeClassifier(), 'decision_tree_classifier')
    print('Naive Bayes Gaussian Classifier...')
    scores['gaussian'] = score_model(naive_bayes.GaussianNB(), 'naive_bayes_gaussian_classifier')
    
    model_scores = pd.DataFrame(scores).mean()
    model_scores.sort_values(ascending=False)
    model_scores.to_csv('model_scores.csv', index=True)
    print('Model scores\n{}'.format(model_scores))

def evaluation():
    print('STEP 4: Evaluation')
    for predicted in predictedValues:
        confusionMatrix(predicted)
    probabilities()
 
############################################################################## 

def removeConstantColumns(train):
    numberOfColumnsBefore = train.columns.shape[0]
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    train.drop(remove, axis=1, inplace=True)
    numberOfColumnsAfter = train.columns.shape[0]
    print("Removed "+str(numberOfColumnsBefore-numberOfColumnsAfter)+" constant columns.")
    
def removeDuplicatedColumns(train):
    numberOfColumnsBefore = train.columns.shape[0]
    remove = []
    cols = train.columns
    for i in range(len(cols)-1):
        v = train[cols[i]].values
        for j in range(i+1,len(cols)):
            if np.array_equal(v,train[cols[j]].values):
                remove.append(cols[j])
    train.drop(remove, axis=1, inplace=True)
    numberOfColumnsAfter = train.columns.shape[0]
    print("Removed "+str(numberOfColumnsBefore-numberOfColumnsAfter)+" duplicated columns.")
    
def deleteColumnsWithHighCorrelation(data):
    max_cor = 0.8
    numberOfColumnsBefore = train.columns.shape[0]
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > max_cor)]
    for col in to_drop:
        del data[col]
    numberOfColumnsAfter = train.columns.shape[0]
    print("Removed "+str(numberOfColumnsBefore-numberOfColumnsAfter)+" columns with correlation > "+str(max_cor)+".")

def removeRowsMissingValues():
    numberOfRowsBefore = train.shape[0]
    for index, row in train.iterrows():
        for column in train:
            if row[column] == -999999 or row[column] == 999999 or row[column] == 9999999999 or row[column] == -9999999999:
                train.drop(index, inplace = True)
                break
    numberOfRowsAfter = train.shape[0]
    print("Removed "+str(numberOfRowsBefore-numberOfRowsAfter)+" rows with missing values.")

def featureSelection():    
    if featureSelectionType == "chi":
        chiSquared()
    if featureSelectionType == "decision":
        decisionTree()

def decisionTree():
    print('Feature selection with decision tree.')
    global test
    global train

    numberOfFeatures = 20
    clf = ExtraTreesClassifier(random_state=1729)
    selector = clf.fit(X_train, y_train)
    feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)
    importantFeatures = feat_imp[:numberOfFeatures]
    print('Selected Features ('+str(numberOfFeatures)+') are:\n'+str(importantFeatures))
    
    train = train[importantFeatures.index.tolist()+['TARGET']]
    test = test[importantFeatures.index.tolist()]

def chiSquared():
    print('Feature selection with chi squared...')
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
    
def dataVisualization():
    ## Heatmap visualization of correlations
    sns.heatmap(train.corr())
    plt.savefig('./Generated_Visualization/heatmap.png')
    i = 0
    for column in train:
        if len(train[column].unique()) < 10 and column != "TARGET":
            sns.countplot(train[column])
            plt.savefig('./Generated_Visualization/column'+str(i)+'.png')
            i = i + 1
    
def confusionMatrix(y_pred):
    y_actu = train['TARGET'].values
    confusion_matrix(y_actu, y_pred[1])    
    y_actu = pd.Series(y_actu, name='Actual')
    y_pred_data = pd.Series(y_pred[1], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred_data, rownames=['Actual'], colnames=['Predicted'], margins=True)
    plot_confusion_matrix(df_confusion, y_pred[0])
    #df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    #plot_confusion_matrix(df_conf_norm, str(str(y_pred[0])+'normalized'))
    df_confusion = df_confusion.astype(float)
    df_confusion.values[0, 0] = df_confusion.values[0, 0] / df_confusion.values[0, 2]
    df_confusion.values[0, 1] = df_confusion.values[0, 1] / df_confusion.values[0, 2]
    df_confusion.values[1, 0] = df_confusion.values[1, 0] / df_confusion.values[1, 2]
    df_confusion.values[1, 1] = df_confusion.values[1, 1] / df_confusion.values[1, 2]
    plot_confusion_matrix(df_confusion, str(y_pred[0])+'probability')
    print('Generated confusion matrices.')
    
def probabilities():
    import warnings
    warnings.filterwarnings('ignore')
    
    # Use 10 estimators so predictions are all multiples of 0.1
    pred_prob = run_prob_cv(train.drop(["TARGET"],axis=1), train.TARGET.values, RF, n_estimators=50)
    pred_churn = pred_prob[:,1]
    is_churn = train.TARGET.values == 1
    
    # Number of times a predicted probability is assigned to an observation
    counts = pd.value_counts(pred_churn)
    
    # calculate true probabilities
    true_prob = {}
    for prob in counts.index:
        true_prob[prob] = np.mean(is_churn[pred_churn == prob])
        true_prob = pd.Series(true_prob)
    
    # pandas-fu
    counts = pd.concat([counts,true_prob], axis=1).reset_index()
    counts.columns = ['pred_prob', 'count', 'true_prob']
    printToCSVWithFilename(counts, 'probabilities_Random_Forest.csv')
    print('Generated Table with probabilities.')
    
##############################################################################
    
def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob
    
def plot_confusion_matrix(df_confusion, title):
    cmap='rainbow'
    plt.matshow(df_confusion, cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    # Loop over data dimensions and create text annotations.
    for i in range(len(df_confusion.columns)):
        for j in range(len(df_confusion.index)):
            text = plt.text(j, i, f'{df_confusion.values[i, j]:.3f}',
                           ha="center", va="center", color="w")
    plt.savefig('./Generated_Visualization/confusionMatrix'+str(title)+'.png')

def splitDataset():
    global X_train
    global X_test
    global y_train
    global y_test
    X_train, X_test, y_train, y_test = train_test_split(train.drop(["TARGET","ID"],axis=1), train.TARGET.values, test_size=0.20, random_state=1729)

def printToCSVWithFilename(data, filename):
    if printToCSV:
        data.to_csv(filename, sep=';', encoding='utf-8')

def histogram(data, x_label, y_label, title):
    _, ax = plt.subplots()
    ax.hist(data, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

def countDistinct():
    dist_counter = 0
    for column in train:
        if len(train[column].unique()) < 2:
            dist_counter = dist_counter + 1
    return dist_counter

def countStrings():
    string_counter = 0
    for column in train:
        for i in train[column]:
            if type(i) is str:
                print(column)
                string_counter = string_counter + 1
    return string_counter

main()
