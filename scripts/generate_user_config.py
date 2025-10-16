#!/usr/bin/env python3
# scripts/generate_user_config.py
import csv
import random

random.seed(42)
with open("csvs/user_config.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user_id","max_delay_ms"])
    for i in range(1, 51):
        user = f"user_{i}"
        # choose a delay between 20 and 200 ms (example)
        delay = random.choice([20, 50, 100, 150, 200])
        w.writerow([user, delay])
print("wrote csvs/user_config.csv")
