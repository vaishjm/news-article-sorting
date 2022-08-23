import joblib
import re
import pandas as pd
import json
from feature_engineering import get_parameter



def model_testing(param):
    # load model, tfidf_path, label_encoder path
    model_path = param['logistic_regression']['save_model']
    tfidf_path = param['load_data']['tfidf']
    label_encoder = param['load_data']['label_encoder']
    # path to save the test report
    test_report_path = param['reports']['test_report']
    # load the testcase
    true_predict_class = param['test_case_true_class']

    # load model
    model = joblib.load(model_path)
    # load tfidf
    tfidf = joblib.load(tfidf_path)
    # load label_encoder
    label_encoder = joblib.load(label_encoder)


    # predict news text
    test_report = {}
    with open(test_report_path, 'w') as data:
        for i in range(1,8):
            test_data = param['test_case']['case' + str(i)]

            X_test = tfidf.transform([test_data])
            predict_score = model.predict(X_test)
            print("predict_score",predict_score[0])
            predict_prob = model.predict_proba(X_test)[0]
            print(predict_prob)
            prediction = label_encoder.inverse_transform(predict_score)

            probability = predict_prob.max()*100
            print("Prabability of News predicting:",prediction[0])
            print("The conditional probability is: %a" %probability)
            case = {
            'case'+str(i): {'true_class': true_predict_class[i-1],
                            'predict_class': prediction[0],
                            'predict_probability': probability
                            }
            }
            test_report.update(case)
        json.dump(test_report, data, indent=4)

if __name__ == "__main__":
    path = "params.yaml"
    param = get_parameter(path)
    model_testing(param)
   
