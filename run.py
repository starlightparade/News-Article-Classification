import numpy as np
import pandas as pd
from codes.Preprocess import ProcessX, xgboost, Gradientboosting, BernoulliNaive, trysvm, randomforest, MultinomialNaive, XGB_Classifier
import xgboost as xgb


class run():

    def run(self):
        print("********* start run***********")
        processX = ProcessX()
        processX.load_data("data/train_v2.csv", "data/test_v2.csv", 0.1)
        x_train = processX.x_train
        x_val = processX.x_val
        y_train = processX.y_train
        y_val = processX.y_val
        x_test = processX.x_test
        x_train_combined = processX.pre_process(x_train=x_train, mode='train')
        x_val_combined = processX.pre_process(x_train=x_val, mode='val')
        x_test_combined = processX.pre_process(x_train=x_test, mode='test')

        # Training XGboost
        xgboost_model = xgboost(x_train_combined, y_train, x_val_combined, y_val)

        randomforest(x_train_combined, y_train, x_val_combined, y_val)
        # SVM
        svm_model = trysvm(x_train_combined, y_train.astype('int'), x_val_combined, y_val.astype('int'))

        x_train_combined = x_train_combined.toarray()
        x_val_combined = x_val_combined.toarray()

        # Bernoulli Naive Bayes
        BNB_model = BernoulliNaive(x_train_combined, y_train.astype('int'), x_val_combined, y_val.astype('int'))

        # Gradient Boosting
        GB_model = Gradientboosting(x_train_combined, y_train.astype('int'), x_val_combined, y_val.astype('int'))

        # Multinomial Naive Bayes
        MNB_model = MultinomialNaive(x_train_combined, y_train.astype('int'), x_val_combined, y_val.astype('int'))
        
        # XGBClassifier
        XGB_Clf_model = XGB_Classifier(x_train_combined, y_train.astype('int'), x_val_combined, y_val.astype('int'))
        
        ########### predict test data ######################
        ########### we have chosen XGB Classifier ##########

        y_predict = XGB_Clf_model.predict(x_test_combined.toarray())
        id = np.arange(len(y_predict))
        y_predict = np.concatenate((id.reshape(-1,1),y_predict.reshape(-1,1)),axis=1)
        df = pd.DataFrame(y_predict, columns=['article_id','category'])
        df.to_csv('Result.csv', index=False)
        print("********* finish run***********")
