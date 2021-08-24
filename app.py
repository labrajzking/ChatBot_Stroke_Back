#import files
from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import pickle#Initialize the flask App
import numpy as np
app = Flask(__name__)
CORS(app,resources={r"/api/*":{"origins":"*"}})
app.config['CORS HEADERS'] = 'Content-Type'
bot = ChatBot('ChatterBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90
        }
    ],
)
#trainer = ChatterBotCorpusTrainer(bot)
#trainer.train("chatterbot.corpus.english")
training_data = open(r"C:\Users\oussema\Desktop\python\flask\one\data\data.txt").read().splitlines()
print (training_data)
iterator=iter(training_data)
trainer=ListTrainer(bot)
trainer.train(training_data)
model = pickle.load(open('model.pkl', 'rb'))
user_info=list()
@app.route("/")
@cross_origin()
def home():    
    return render_template("home.html") 
@app.route("/user",methods=["POST"])
@cross_origin()
def get_bot_response():   
    try:
        userText = request.json
        data=userText['msg']   
        user_info.append(data)
        print (user_info)
        return next(iterator)
        #return str(bot.get_response(userText)) 
    except StopIteration:
        # exception will happen when iteration will over
        probas=model.predict_proba([np.array(user_info[1:])])
        print(probas)
        return render_template('results.html', prediction_text=' Your Stroke Percentage is :{}'.format(probas))
if __name__ == "__main__":    
    app.run()