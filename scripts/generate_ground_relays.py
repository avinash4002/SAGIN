#!/usr/bin/env python3
# scripts/generate_ground_relays.py
import csv

relays = {
    "gbs_1": (1000.0, 1000.0, 25.0),
    "gbs_2": (1000.0, 9000.0, 25.0),
    "gbs_3": (9000.0, 1000.0, 25.0),
    "gbs_4": (9000.0, 9000.0, 25.0),
    "gbs_5": (5000.0, 5000.0, 30.0),
}

with open("csvs/ground_relay_mobility.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestep","relay_id","x","y","z"])
    for t in range(1, 3601):
        for rid, (x,y,z) in relays.items():
            w.writerow([t, rid, x, y, z])
print("wrote csvs/ground_relay_mobility.csv")
