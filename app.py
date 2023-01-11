from flask import Flask, request, jsonify, redirect, render_template
from main import cleaning, load_trained_models, lets_predict
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    user_input = request.form['news_article']
    print(type(user_input))
    # user_input = data
    label_encoder, trained_transformer_model, trained_ml_model = load_trained_models()
    prediction = check_prediction(user_input=user_input, trained_ml_model=trained_ml_model, trained_transformer_model=trained_transformer_model)
    # Return the result
    # label_encoder.inverse_transform(prediction)
    return jsonify({'category': label_encoder.inverse_transform(prediction).tolist()})



def check_prediction(user_input, trained_ml_model, trained_transformer_model):
    # user_input = user_input.apply(cleaning)
    prediction = lets_predict(lr_model=trained_ml_model, trained_vectorizer= trained_transformer_model, fresh_data= user_input)
    print("after prediction")
    # return jsonify({'category': str(prediction)})
    return prediction




if __name__ == '__main__':
    app.run(debug=True)



# sample = pd.DataFrame(["Months before the assembly election in Karnataka, Bharatiya Janata Partys (BJP) senior leader and the state former Chief Minister SM Krishna announced his retirement from active politics on Wednesday"],columns=['Text'])
#
# print(check_prediction(sample['Text']))