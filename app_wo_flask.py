# from flask import Flask, request, jsonify
from main import cleaning, load_trained_models, lets_predict
import pickle
import pandas as pd

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    # user_input = request.form['data']
    user_input = "Months before the assembly election in Karnataka, Bharatiya Janata Partys (BJP) senior leader and the state former Chief Minister SM Krishna announced his retirement from active politics on Wednesday"
    print(type(user_input))
    # user_input = data
    label_encoder, trained_transformer_model, trained_ml_model = load_trained_models()
    prediction = check_prediction(user_input=user_input, trained_ml_model=trained_ml_model, trained_transformer_model=trained_transformer_model)
    print(f'prediction {prediction}')
    # Return the result
    # label_encoder.inverse_transform(prediction)
    return label_encoder.inverse_transform(prediction)

def check_prediction(user_input, trained_ml_model, trained_transformer_model):
    # user_input = user_input.apply(cleaning)
    prediction = lets_predict(lr_model=trained_ml_model, trained_vectorizer= trained_transformer_model, fresh_data= user_input)
    print("after prediction")
    # return jsonify({'category': str(prediction)})
    return prediction



print(predict())
# sample = pd.DataFrame(["Months before the assembly election in Karnataka, Bharatiya Janata Partys (BJP) senior leader and the state former Chief Minister SM Krishna announced his retirement from active politics on Wednesday"],columns=['Text'])
#
# print(check_prediction(sample['Text']))