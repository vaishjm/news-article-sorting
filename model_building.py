import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from feature_engineering import get_parameter
import json


def model_creation(X, y, pr):
    """
    Description: train the model
    return: classifier model
    """

    c = pr['C']
    p_n = pr['penalty']
    s = pr['solver']
    c_w = pr['class_weight']
    m_c = pr['multi_class']
    lr_model = LogisticRegression(C = c,
                              class_weight=c_w,
                              multi_class=m_c,
                                penalty = p_n,
                                solver = s)
    
    lr_model.fit(X, y)
    return lr_model


def report(m_score, train_accuracy, test_accuracy):
    """
    Description: save the score for model
    return:
    """
    with open(m_score, 'w') as data:
        score = {
            "Training Accuracy:": train_accuracy,
            "Testing Accuracy:": test_accuracy,
        }
        json.dump(score, data, indent=4)


def model_build(parameteras):
    # load model parameter
    params = parameteras['logistic_regression']['params']
    # load train and test data
    X_train_path = parameteras['load_data']['X_train']
    X_test_path = parameteras['load_data']['X_test']
    y_train_path = parameteras['load_data']['y_train']
    y_test_path = parameteras['load_data']['y_test']
    # save the model
    save_model_path = parameteras['logistic_regression']['save_model']
    # m_params = parameteras['reports']['parameters']
    m_score = parameteras['reports']['scores']


    # load train and test data
    y_train = joblib.load(y_train_path)
    X_train = joblib.load(X_train_path)
    X_test = joblib.load(X_test_path)
    y_test = joblib.load(y_test_path)


    # model creation
    model = model_creation(X_train, y_train, params)
    joblib.dump(model, save_model_path)

    # predict test data
    test_predict_data = model.predict(X_test)
    print('test_pred_data', test_predict_data)
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_predict_data)
    print("training accuracy:", train_accuracy)
    print("testing accuracy:", test_accuracy)
    # classification_report = classification_report(y_test, test_predict_data)
    # confusion_matrix = confusion_matrix(y_test, test_predict_data)
    # print("classification_report:", classification_report)
    # print("confusion_matrix:", confusion_matrix)

    # create report
    report(m_score, train_accuracy, test_accuracy)





if __name__ == "__main__":
    path = "params.yaml"
    parameteras = get_parameter(path)
    model_build(parameteras)

    
