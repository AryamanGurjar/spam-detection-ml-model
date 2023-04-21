from flask import Flask, render_template, request
import pickle
import pandas as pd

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
        f = request.files['file']
        df = pd.read_excel(f)
        messages = df['message'].tolist()
        # Convert the text to a numerical representation
        messages = df['message'].astype(str).dropna().tolist()
        messages = cv.transform(messages)

        predictions = model.predict(messages)
        # Add the predictions as a new column in the dataframe
        df['prediction'] = predictions
        # Convert the dataframe to HTML
        result_html = df.to_html()
        # Render the result template and pass the result HTML as a parameter
        return render_template('result.html', result_html=result_html)

if __name__ == '__main__':
    app.run(debug=True)
