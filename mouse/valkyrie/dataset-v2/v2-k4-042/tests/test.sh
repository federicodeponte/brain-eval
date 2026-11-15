#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/grade.py "/app/answer.txt" /tests/expected.json /logs/verifier
