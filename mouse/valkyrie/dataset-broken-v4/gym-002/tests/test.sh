#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/grade.py /app/output/predictions.json /tests/key.json /logs/verifier
