from flask import Flask, render_template, request
import joblib
# from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.base import BaseEstimator, TransformerMixin

def preprocess_text( text):
    # Remove punctuations, digits, and other non-Arabic characters
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_texts(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    return sequences, word_index, tokenizer 

def pad_texts(sequences, maxlen=45):
    return pad_sequences(sequences, maxlen=maxlen,padding='post' ,truncating='post')

class TextPreprocessor(BaseEstimator, TransformerMixin):


    # Apply preprocessing to the entire text column
    def preprocess_data(self , df):
        return df['text'].apply(preprocess_text)
        



    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.preprocess_data(X)
        sequences, word_index, tokenizer  = tokenize_texts(X)
        paded_text = pad_texts(sequences, maxlen=45)        
        return paded_text


  
        

# Load the model from the file
loaded_model = joblib.load(r'F:\projects iti\nlp\NLP-Project\Deployment NLP\pipe.joblib')

LSTM_model =  joblib.load(r'C:\Users\Hendy Group\Desktop\NLP\project\LSTM_pipeline.joblib')


# Use the loaded model to make predictions or evaluate
dilect_dict= {0: "لهجة مصرية",
              1: "لهجة لبنانية",
              2: "لهجة ليبية",
              3: "لهجة مغربية" ,
              4: "لهجة سودانية"  }


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text_input')
        select = request.form.get('model_select')


        print(select)
        
        if text:
            if select == '1':
                text = [text]
                text = loaded_model.predict(text)
                return render_template('index.html' , result=dilect_dict[ int(text[0]) ])
            elif select == '2':

                test = pd.DataFrame({text})
                text = test.rename(columns={0:'text'})
                text = LSTM_model.predict(text)
                return render_template('index.html' , result=dilect_dict[ np.argmax(text)])
                
        print(text)

        # proseing
        # summarization
        return render_template('index.html' , result=dilect_dict[ int(text[0]) ])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
