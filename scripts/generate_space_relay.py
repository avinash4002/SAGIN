#!/usr/bin/env python3
# scripts/generate_space_relay.py
import csv
import math
import sys

ENTRY = (0.0, 8000.0)
EXIT  = (10000.0, 2000.0)
ALT   = 550000.0
GROUND_SPEED = 7500.0  # m/s

def interpolate_over_time(entry, exit, t, T):
    ex, ey = entry
    sx, sy = exit
    ratio = t / max(T-1,1)
    x = ex + (sx-ex)*ratio
    y = ey + (sy-ey)*ratio
    return x,y

def realistic_path(entry, exit, t):
    # compute vector velocity from entry toward exit at GROUND_SPEED,
    # but we will let the satellite be at entry at t=0 and move with GROUND_SPEED
    ex, ey = entry
    sx, sy = exit
    dx = sx - ex
    dy = sy - ey
    dist = math.hypot(dx,dy)
    if dist == 0:
        return ex,ey
    ux, uy = dx/dist, dy/dist
    travelled = GROUND_SPEED * t
    x = ex + ux*travelled
    y = ey + uy*travelled
    return x,y

mode = "interpolate"   # default
if len(sys.argv) > 1 and sys.argv[1] in ("interpolate", "realistic"):
    mode = sys.argv[1]

with open("csvs/space_relay_mobility.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestep","relay_id","x","y","z"])
    for t in range(3600):
        if mode == "interpolate":
            x,y = interpolate_over_time(ENTRY, EXIT, t, 3600)
        else:
            x,y = realistic_path(ENTRY, EXIT, t)
        w.writerow([t+1, "leo_1", float(x), float(y), float(ALT)])
print("wrote csvs/space_relay_mobility.csv with mode:", mode)
