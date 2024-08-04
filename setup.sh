#!/bin/bash
# Setup and activate virtual environment 
python3 -m venv env
source env/bin/activate

# Install requirements
pip install -r requirements.txt