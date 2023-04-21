from flask import Flask, render_template, request
import pickle
# from sklearn.feature_extraction.text import CountVectorizer

# Load the vectorizer and the model
cv = pickle.load(open('cv.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = [message]
        # print(prediction)
        # # Convert the text to a numerical representation
        vector = cv.transform(prediction).toarray()
        # # Make the prediction
        my_pred = model.predict(vector)
    return render_template('result.html', prediction=my_pred)

if __name__ == '__main__':
    app.run(debug=True)
