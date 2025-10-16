#!/usr/bin/env python3
# scripts/generate_air_relays.py
import csv

def pos_on_rectangle(minx, miny, w, h, dist):
    sides = [w, h, w, h]
    pts = [(minx, miny), (minx+w, miny), (minx+w, miny+h), (minx, miny+h)]
    perim = 2*(w+h)
    if perim == 0:
        return minx, miny
    dist = dist % perim
    for i, side in enumerate(sides):
        if dist <= side:
            x0, y0 = pts[i]
            x1, y1 = pts[(i+1)%4]
            ratio = dist/side if side > 0 else 0
            x = x0 + (x1-x0)*ratio
            y = y0 + (y1-y0)*ratio
            return x, y
        dist -= side
    return pts[0]

# UAV definitions (minx, miny, width, height, altitude, speed)
uavs = {
    "uav_1": (6000.0, 6000.0, 4000.0, 4000.0, 150.0, 30.0),
    "uav_2": (1000.0, 1000.0, 4000.0, 4000.0, 180.0, 35.0),
    "uav_3": (1000.0, 1000.0, 8000.0, 8000.0, 250.0, 50.0),
}

with open("csvs/air_relay_mobility.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestep", "relay_id", "x", "y", "z"])
    for t in range(3600):  # t from 0..3599
        for rid, params in uavs.items():
            minx, miny, width, height, alt, speed = params
            dist = speed * t  # meters travelled along rectangle perimeter
            x, y = pos_on_rectangle(minx, miny, width, height, dist)
            writer.writerow([t+1, rid, float(x), float(y), float(alt)])
print("wrote csvs/air_relay_mobility.csv")
