from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix, hstack
import re
import xgboost as xgb
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score


class ProcessX():
    def __init__(self):
        self.label_publisher = None
        self.label_hostname = None
        self.vectorizer = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None

    def load_data(self, train_filepath, test_filepath, val):
        x_y_train = pd.read_csv(train_filepath, encoding="ISO-8859-1").as_matrix()
        print(x_y_train.shape)

        self.x_test = pd.read_csv(test_filepath, encoding="ISO-8859-1").as_matrix()
        N, features = x_y_train.shape
        x = x_y_train[:, 0:features - 1]
        y = x_y_train[:, features - 1]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=val, random_state=42)

        # get column of publisher and hostname from all the data
        x_feature_publisher = np.concatenate((self.x_train[:, 3], self.x_val[:, 3]), axis=0)
        x_feature_publisher = np.concatenate((x_feature_publisher, self.x_test[:, 3]), axis=0)
        x_feature_hostname = np.concatenate((self.x_train[:, 4], self.x_val[:, 4]), axis=0)
        x_feature_hostname = np.concatenate((x_feature_hostname, self.x_test[:, 4]), axis=0)
        # initialize LabelEncoder
        self.label_publisher = preprocessing.LabelEncoder()
        self.label_publisher.fit(x_feature_publisher.astype(str))

        self.label_hostname = preprocessing.LabelEncoder()
        self.label_hostname.fit(x_feature_hostname.astype(str))

    def pre_process(self, x_train, mode='train'):

        new_documents = []
        titles = [re.sub(r'[^a-zA-Z ]', r'', str(s)).lower() for s in x_train[:, 1]]
        #         contents = [re.sub(r'[^a-zA-Z ]',r'',str(s)) for s in x_train[:,2]]

        documents = titles
        spell = SpellChecker()
        stop_words = set(stopwords.words('english'))
        wordnet_lemmatizer = WordNetLemmatizer()

        for x in documents:
            x_str = str(x)

            tokens = word_tokenize(x_str)
            new_tokens = [w for w in tokens if not w in stop_words and not w.isdigit()]
            misspelled = spell.unknown(new_tokens)
            new_tokens = [w for w in new_tokens if not w in misspelled or (w in misspelled and len(w) > 3)]
            document = [wordnet_lemmatizer.lemmatize(token) for token in new_tokens]
            new_documents.append(document)

        X1 = x_train[:, 3]
        X2 = x_train[:, 4]

        ################## initialize vectorizer if in training mode #################
        ################## get sparse matrix of tf-idf vectors for training and test data #################

        if mode == 'train':
            with open('stopwords.txt') as f:
                lines = f.readlines()
                customized_stopwords = [line.strip().replace("\n", "") for line in lines]
            self.vectorizer = TfidfVectorizer(stop_words=customized_stopwords, analyzer='word', tokenizer=process_token,
                                              preprocessor=process_token, token_pattern=None, max_features=20000)
            X_tfidf = self.vectorizer.fit_transform(new_documents)

        elif self.vectorizer:
            X_tfidf = self.vectorizer.transform(new_documents)

        #################3. encode feature publisher and domain with labels###############

        X_1_processed = self.label_publisher.transform(X1.astype(str))

        ##################  convert to sparse matrix  #################
        X_1_processed_sparse = sparse.csr_matrix(X_1_processed.reshape(-1, 1))

        X_2_processed = self.label_hostname.transform(X2.astype(str))
        X_2_processed_sparse = sparse.csr_matrix(X_2_processed.reshape(-1, 1))

        ################# #combine publisher and domain with tf-idf sparse matrix#################
        x_train_combined = hstack((X_tfidf, X_1_processed_sparse))
        x_train_combined = hstack((x_train_combined, X_2_processed_sparse))

        return x_train_combined


def process_token(doc):
    return doc


def xgboost(x_train_combined, y_train, x_val_combined, y_val):
    dtrain = xgb.DMatrix(x_train_combined, label=y_train.astype('int'))
    dval = xgb.DMatrix(x_val_combined)

    params = {'max_depth': 3, 'min_child_weight': 1, 'eta': 0.1, 'silent': 1, 'objective': 'multi:softmax',
              'num_class': 5,
              'eval_metric': 'mlogloss'}
    num_boost_round = 699
    bst = xgb.train(params, dtrain,
                    num_boost_round)  # ,evals= [(dval1,'eval'), (dtrain,'train')], early_stopping_rounds=5

    y_pred = bst.predict(dval)

    accuracy = accuracy_score(y_val.astype(int), y_pred.astype(int))
    print("XGBoost Accuracy: %.20f%%" % (accuracy * 100.0))
    return bst


def MultinomialNaive(x_train, y_train, x_val, y_val):
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    #   print(y_val, y_pred)
    accuracy = accuracy_score(y_val.astype(int), y_pred.astype(int))
    print("MultinomialNB Accuracy: %.2f%%" % (accuracy * 100.0))

    return model


def BernoulliNaive(x_train, y_train, x_val, y_val):
    model = BernoulliNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    #   print(y_val, y_pred)
    accuracy = accuracy_score(y_val.astype(int), y_pred.astype(int))
    print("BernoulliNB Accuracy: %.2f%%" % (accuracy * 100.0))

    return model


def Gradientboosting(x_train_combined, y_train, x_val_combined, y_val):
    gb = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=10, subsample=1.0)
    print(x_train_combined.shape, y_train.shape)
    gb.fit(x_train_combined, y_train.astype(int))
    y_pred = gb.predict(x_val_combined)
    #   print(y_val, y_pred)
    accuracy = accuracy_score(y_val.astype(int), y_pred.astype(int))
    print("Grandient boosting Accuracy: %.2f%%" % (accuracy * 100.0))
    return gb


def trysvm(x_train_combined, y_train, x_val_combined, y_val):
    clf = svm.SVC(kernel='rbf', C=1e3, gamma=0.1)  # gamma='scale',decision_function_shape='ovo'
    clf.fit(x_train_combined, y_train.astype(int))
    y_pred = clf.predict(x_val_combined)
    accuracy = accuracy_score(y_val.astype(int), y_pred.astype(int))
    print("SVM Accuracy: %.2f%%" % (accuracy * 100.0))

    return clf


def randomforest(x_train_processed, y_train, x_val_processed, y_val):
    model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                          random_state=0)

    model.fit(x_train_processed, y_train.astype('int'))

    # validation
    score = model.score(x_val_processed, y_val.astype('int'))
    print("Random Forest Accuracy: %.2f%%" % (score * 100.0))
    return model

def XGB_Classifier(x_train_processed, y_train, x_val_processed, y_val):
    model = xgb.XGBClassifier(n_estimators=500, nthread=-1, 
                         max_depth = 5, seed=0, objective='multi:softprob', num_class=5, learning_rate=0.1,
                                subsample=0.7,min_child_weight=0.05,colsample_bytree=0.3)
    model.fit(x_train_processed.toarray(), y_train.astype('int'), eval_metric= 'merror', 
         eval_set=[ (x_val_processed.toarray(), y_val.astype('int'))],verbose=True, early_stopping_rounds=15)
    score = accuracy_score(y_val.astype(int), model.predict(x_val_combined.toarray()).astype(int))
    # validation
    print("xgbClassifer Accuracy: %.2f%%" % (score * 100.0))
    return model
