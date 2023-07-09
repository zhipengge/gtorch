# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20230614
@file: app.py
@brief: app
"""
from flask import Flask, render_template
from functools import wraps
import os
import config
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import utils

acvite = "index"
file_tree = {}

def update_active(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global active
        active = func.__name__
        return func(*args, **kwargs)
    return wrapper

def update_output_file_tree(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global file_tree
        file_tree = utils.get_file_tree(config.output_dir)
        return func(*args, **kwargs)
    return wrapper

app = Flask(__name__)
app.config.from_object(config)

@app.route("/")
@update_active
def index():
    return render_template("index.html", active=active)

@app.route("/analyse/")
@update_active
@update_output_file_tree
def analyse():
    return render_template("analyse.html", active=active, file_tree=file_tree)

if __name__ == "__main__":
    app.run(debug=True, port=8000, host=config.HOST)