# -*- coding: utf-8 -*-

from flask import Flask,jsonify,render_template,request
from mnist import Trainer
import numpy as np
import json
app = Flask(__name__)

trainer = None


@app.route('/')
def trainmnist():
    return render_template("index.html")

@app.route('/prepare',methods=['POST'])
def prepare():
    jsondata = request.get_data()
    paradict = json.loads(jsondata)
    global trainer
    trainer = Trainer(paradict)
    return jsonify(code=200)

@app.route('/start_train',methods=['POST'])
def start_train():
    trainer.train_epoch()
    return jsonify({"code":200})

@app.route('/get_loss',methods=['GET'])
def get_loss():
    if trainer is not None:
        x, ty, vy = trainer.get_loss()
        traindata = dict(
            x=x,
            y=ty,
            type='plot',
            mode='lines',
            marker=dict(
                color='black',
            )
        )
        valdata = dict(
            x=x,
            y=vy,
            type='plot',
            mode='lines',
            marker = dict(
                color='blue',
            )
        )

    else:
        traindata = dict(
            x=[],
            y=[],
            type='plot'
        )
        valdata = dict(
            x=[],
            y=[],
            type='plot'
        )
    return jsonify(Traindata=traindata,Valdata=valdata)

@app.route('/get_predict',methods=['GET'])
def get_predict():
    if trainer is not None:
        predimg,predlable=trainer.get_pred_mnist()
        return jsonify(Code="200",Predimg=predimg,Predlabel=predlable)
    else:
        return jsonify(Code="404")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=True)
