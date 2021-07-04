from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import nltk
nltk.download('stopwords')
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os
stemmer=PorterStemmer()
from utils import get_base_url, allowed_file, and_syntax
from simple_version import get_summary


# setup the webservver
# port = 2022
# base_url = get_base_url(port)
app = Flask(__name__)


IMAGE_FOLDER=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=IMAGE_FOLDER
    


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result',methods=["GET","POST"])
@cross_origin()
def result():
	if request.method=="POST":
		word_list=[]
		link=(request.form["Link"])
		
		output= get_summary(link)

		return render_template('home.html', prediction_text=output)

	return render_template("home.html")



# if __name__ == "__main__":
    # change the code.ai-camp.org to the site where you are editing this file.
#    print("Try to open\n\n    https://cocalc3.ai-camp.org" + base_url + '\n\n')
    # remove debug=True when deploying it
#    app.run(host = '0.0.0.0', port=port, debug=True)
#    import sys; sys.exit(0)


