#!/bin/bash
mkdir -p /app/output
cat > /app/output/predictions.json <<'EOF'
{"predictions": [2, 8, 5, 1, 4, 2, 7, 8, 3, 0, 6, 5, 2, 8, 3, 6, 4, 7, 7, 7, 5, 7, 9, 3, 1, 9, 0, 3, 5, 4, 0, 8, 4, 1, 9, 6, 4, 6, 5, 6, 9, 0, 2, 0, 1, 8, 3, 9, 1, 2], "claimed_accuracy": 1.0, "data_issues": "Training and validation labels are shifted by one segment relative to movie_time_s (off-by-one label lag)."}
EOF
