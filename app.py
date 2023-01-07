from flask import Flask, request, jsonify
from main import cleaning, lets_predict
import pickle

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    data = request.args.get('data')
    user_input = data['text']

    # Preprocess the input
    user_input = user_input.apply(cleaning)

    trained_model = load_model()

    prediction = lets_predict(trained_model, user_input)
    # Return the result
    return jsonify({'category': prediction})



def load_model():
    trained_model = pickle.load(open(r'D:\ML\\article_sort\model\\trained_model.pkl', 'rb'))
    return trained_model


if __name__ == '__main__':
    app.run(debug=True)