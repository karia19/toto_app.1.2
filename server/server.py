#from crypt import methods
from urllib import response
from flask import Flask, send_file, make_response, render_template
from flask import request
from flask_cors import CORS
import json
import pandas as pd
#import city_stats
import today
#import machine_learn_server
import redis_python
import main_logastic
import pickle

app = Flask(__name__)
CORS(app)

@app.route('/api/v1/toto/', methods=['GET', 'POST'])
def test():
    if request.method == "GET":
        return {"message": "this path work"}

@app.route('/api/v1/toto/odds', methods=['GET', 'POST'])
def main():

    if request.method == "GET":
        #res_city = city_stats.search_y_city("Seinäjoki")
        
        #return json.dumps(res_city)
        return {"message": "hello daa"}

    if request.method == "POST":
        res = request.get_json()
        res_from_redis = redis_python.get_only_data(res['city'])
        #print(res_from_redis)

        return res_from_redis
        



@app.route('/api/v1/toto/today', methods=['GET', 'POST'])
def horses_today():
        
    if request.method == "POST":
        res = request.get_json()
        res_roday = today.main_today(res['city'])
        redis_python.save_data_redis(res['city'])

        #return { "message": "working on progress" }
        return json.dumps(res_roday)


@app.route('/api/v1/toto/history', methods=['GET', 'POST'])
def history():
    if request.method == "POST":
        res = request.get_json()

        try:
            with open('/app/track_stats/' + res['city'] + '.pkl', 'rb') as handle:
                b = pickle.load(handle)
            #res_from_redis = redis_python.get_only_data(res['city'])
            return b
        except:
            return {"message": "no data"}


@app.route("/api/v1/toto/machine_learn", methods=['GET', 'POST'])
def machine():

    if request.method == "POST":
        res = request.get_json()
        machine_res = main_logastic.get_collection(res['city'])

        return machine_res


@app.route("/api/v1/toto/history_ods", methods=["GET", "POST"])
def history_ods():

    if request.method == "POST":
        try:
            res = request.get_json()
            res_history_ods = redis_python.get_history_ods(res['day'])
            return json.dumps(res_history_ods)  
        except:
            return { "message": "not valid"}





if __name__=='__main__':
    app.run(debug=False, port=8000)
