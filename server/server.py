from flask import Flask, jsonify, request
import pandas as pd
import joblib
import ktrain
from ktrain import text

from keras import initializers

app = Flask(__name__)

@app.route("/", methods=['POST'])
def do_prediction():
    ##json = request.get_json()
    ##text = request.text()
    text = request.get_json()
    print(text)

    ##print('**Load Saved Model and Predict**')
    predictor1 = ktrain.load_predictor('../content/bert_model_Suicide')
    
    ##data = "I'm so tired of pretending that everything is okay. I just want it to be over"
    ##print(predictor1.predict(data))

    ##result = {"Predicted House Price" : y_predict[0]}
    #print(result)
    ##return jsonify(predictor1.predict(data))
    return jsonify(predictor1.predict(text))

if __name__ == "__main__":
     app.run(debug=True, host = '127.0.0.1', port = 8000)
