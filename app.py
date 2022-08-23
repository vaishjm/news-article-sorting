import json
from flask import Flask, render_template, request, jsonify, Response, url_for, redirect
from flask_cors import CORS, cross_origin
import yaml
import joblib
import logging
import time
from data_processing import lemmatize, remove_special_characters


# logging basic configuration
logging.basicConfig(filename="logging\\test.txt",
                    filemode='a',
                    format='%(asctime)s %(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG
                    )


logging.info('News Articles Sorting ready for use!')



# Load the parameters.
path = "params.yaml"

with open(path, encoding='UTF-8') as data:
    parameters = yaml.safe_load(data)
    logging.info("Parameters loaded successfully.")
model_path = parameters['logistic_regression']['save_model']
tfidf_path = parameters['load_data']['tfidf']
label_encoder = parameters['load_data']['label_encoder']

try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully!.")

    tfidf = joblib.load(tfidf_path)
    logging.info("tfidf vectorization loaded successfully!.")

    label_encoder = joblib.load(label_encoder)
    logging.info("label_encoder loaded successfully!.")
except Exception as ex:
    logging.exception(ex)



# initialising the flask app with the name 'app'
app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
@cross_origin()
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def predict():
    if request.method == 'POST':
        # Obtaining the Text entered
        text = request.form['data']
        start_time = time.time()
        print(start_time)
        try:
            if text:
                #text = text_preprocess(text)
                prediction, predict_prob, probability = model_predict(text)
                print(prediction)
                print(predict_prob)
                end_time = time.time() - start_time
                print(end_time)
                result_final = json.dumps({'category': prediction[0], 'probability': probability, 'execution_time': end_time,
                                           'y': predict_prob.tolist()})
                logging.info("Output have printed on webpage.")
                return result_final
            else:
                result_final = {'category': "Please! put some text in textbox.", 'probability':0, "execution_time": 0 }
                logging.warning('Please put text in textbox.')
                return result_final

        except Exception as ex:
            logging.exception(ex)
            print("(app.py) - Something went wrong.\n" + str(ex))


    else:

        logging.error("post method in after classify button not working.")
        return render_template('index.html')

"""def text_preprocess(text):
    text = remove_special_characters(text)
    text = text.lower()
    text = lemmatize(text)
    return text
"""
def model_predict(text):
    try:
        x_test = tfidf.transform([text])
        logging.info("Text representation have done.")
        predict = model.predict(x_test)
        logging.info("Model has predicted successfully!.")
        predict_prob = model.predict_proba(x_test)[0]
        probability = round(predict_prob.max() * 100)
        prediction = label_encoder.inverse_transform(predict)

        return prediction, predict_prob, probability

    except Exception as ex:
        logging.exception(ex)



if __name__ == '__main__':
    app.run(debug=True)
