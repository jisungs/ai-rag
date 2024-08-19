print('hello world')

from flask import Flask



@app.route("/")
def home():
    render('index.html')