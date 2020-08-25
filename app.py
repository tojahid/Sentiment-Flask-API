from flask import Flask, request, redirect, render_template
from senttiment_model import *


app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        return render_template("index.html")

    return render_template("index.html")


@app.route('/sentiment_home',methods = ['POST', 'GET'])
def sentiment_home():
    if request.method == 'POST':
        return render_template('sentiment_page.html')
        
   
@app.route('/sentiment_output',methods = ['POST', 'GET'])
def sentiment_output():
    if request.method == 'POST':
        sent_text = request.form['sent']
        if sent_text == "":
            output = "Null"
            sent_text = "Null"
            results = {'text': sent_text, "output":output}
        else:
            output = classify(sent_text)
            results = {'text': sent_text, "output":output}
        print(output)
        print(sent_text)
        return render_template('sentiment_output.html', results=results)




if __name__ == '__main__':
    app.run(debug=True)