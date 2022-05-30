from urllib import response
from flask import Flask, send_file, make_response, render_template
from flask import request
from flask_cors import CORS
import json
import pandas as pd
import city_stats

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/toto', methods=['GET', 'POST'])
def main():

    if request.method == "GET":
        #res_city = city_stats.search_y_city("Seinäjoki")
        
        #return json.dumps(res_city)
        return {"message": "hello daa"}

    if request.method == "POST":
        res = request.get_json()
        res_city = city_stats.search_y_city(res['city'])
        
        return json.dumps(res_city)
        #return {"message": "hello daa"}

if __name__=='__main__':
    app.run(debug=False, port=8000)