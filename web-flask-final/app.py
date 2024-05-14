from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('pkl/random_forest.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('pkl/logistic_regression.pkl', 'rb') as lr_file:
    lr_model = pickle.load(lr_file)

with open('pkl/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route('/')
def home():
    title = 'Home'
    return render_template('index.html', title=title)

@app.route('/random-forest', methods=['GET', 'POST'])
def random_forest():
    title = 'Random Forest'
    url = 'random_forest'
    desc = 'Random forest is a machine learning algorithm that combines the results of multiple decision trees to produce a single result.'
    review = ""
    prediction = None
    probability = None
    error = None
    if request.method == 'POST':
        review = request.form.get('review')
        if len(review) > 15:
            title = 'Result Random Forest'
            transformed_review = vectorizer.transform([review])
            prediction = rf_model.predict(transformed_review)[0]
            probabilities = rf_model.predict_proba(transformed_review)
            probability = probabilities[0][1] * 100
            probability = round(probability, 2)
            return render_template('result.html', title=title, url=url, review=review, prediction=prediction, probability=probability)
        else:
            error = "Minimal 15 characters"
    return render_template('form.html', title=title, url=url, review=review, error=error, desc=desc)

@app.route('/logistic-regression', methods=['GET', 'POST'])
def logistic_regression():
    title = 'Logistic Regression'
    desc = 'Logistic regression is a process of modeling the probability of a discrete outcome given an input variable.'
    url = 'logistic_regression'
    review = ""
    prediction = None
    probability = None
    error = None
    if request.method == 'POST':
        review = request.form.get('review')
        if len(review) > 15:
            title = 'Result Logistic Regression'
            transformed_review = vectorizer.transform([review])
            prediction = lr_model.predict(transformed_review)[0]
            probabilities = lr_model.predict_proba(transformed_review)
            probability = probabilities[0][1] * 100
            probability = round(probability, 2)
            return render_template('result.html', title=title, url=url, review=review, prediction=prediction, probability=probability)
        else:
            error = "Minimal 15 characters"
    return render_template('form.html', title=title, url=url, review=review, error=error, desc=desc)

if __name__ == '__main__':
    app.run(debug=True)
