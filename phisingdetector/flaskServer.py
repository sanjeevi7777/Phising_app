from flask import Flask, request, jsonify
from flask_cors import CORS
# import pickle
# import numpy as np
app = Flask(__name__, instance_relative_config=True)
import phising as fs
# Define your endpoint that processes input data and returns predictions
cors=CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.data
    # print(request.data)
    # print(request)

    return fs.get_prediction_from_url(str(data))
    # data = np.reshape(data, (1, -1))
    # file = open('./model_pickle', 'rb')
    # Process the input data with your trained model
    # model = pickle.load(file)
    # prediction = model.predict(data)

    # Return the prediction as a JSON response
    # return jsonify({'prediction': prediction})
# print(predict())
app.run(debug=True,port=5001)