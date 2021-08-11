from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
import numpy as np
import os
from flask_cors import cross_origin
from utils import get_base_url, allowed_file, and_syntax
from classifier import predict

# setup the webserver
# port = 2025
# base_url = get_base_url(port)
# app = Flask(__name__, static_url_path=base_url+'static')
app = Flask(__name__)

IMAGE_FOLDER=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=IMAGE_FOLDER
    


# @app.route(base_url)
def home():
    return render_template('home.html')


# @app.route(base_url + "result", methods=["GET","POST"])
@cross_origin()
def result():
    if request.method=="POST":
        review = (request.form["Review"])
        prediction = predict(review)
        output = prediction
    return render_template('home.html',prediction_text=f' {output}')
    return render_template("home.html")





if __name__ == "__main__":
    # change the code.ai-camp.org to the site where you are editing this file.
    print("Try to open\n\n    https://cocalc5.ai-camp.org" + base_url + '\n\n')
    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)


