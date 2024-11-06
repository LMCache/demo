#!/bin/bash
set -e

timeout 300 bash -c 'until curl -X POST localhost:8002/v1/completions > /dev/null 2>&1;
    do
    echo "waiting for server to start..."
    sleep 1
    done'

python3 preprocess.py
streamlit run frontend-double.py
