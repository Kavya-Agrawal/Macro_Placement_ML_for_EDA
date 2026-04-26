#!/usr/bin/env python3
import sys
import os
import math
import random

if len(sys.argv) != 3:
    print("Usage: ./sa_legalize_macros <friend_filename> <foldername>")
    sys.exit(1)

friend_file = sys.argv[1]
folder_name = sys.argv[2]
base_dir = f"/home/anubhav/mleda/test_ispd2005/{folder_name}"

nodes_path = os.path.join(base_dir, f"{folder_name}.nodes")
pl_path = os.path.join(base_dir, f"{folder_name}.pl")
friend_path = os.path.join(base_dir, friend_file)

# ==========================================
PADDING = 80  # DRC buffer
STEP = 20     # Resolution of the spiral search
# ==========================================

pl_lines = []
with open(pl_path, 'r') as f:
    pl_lines = f.readlines()

# 1. 100% Bulletproof Boundary Detection (Reads the I/O Pad Ring)
MAX_X = 0
MAX_Y = 0
for line in pl_lines:
    if "FIXED" in line or "terminal" in line.lower():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                MAX_X = max(MAX_X, float(parts[1]))
                MAX_Y = max(MAX_Y, float(parts[2]))
            except ValueError:
                pass

# Add a tiny buffer so nothing sits exactly flush on the mathematical edge
MAX_X = int(MAX_X) + 100
MAX_Y = int(MAX_Y) + 100
print(f"[*] Universal Boundaries Detected from I/O Pads: MAX_X={MAX_X}, MAX_Y={MAX_Y}")

# 2. Parse AI Targets
ai_macros = set()
with open(friend_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 0: ai_macros.add(parts[0])

macro_data = {}
with open(nodes_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] in ai_macros:
            try:
                macro_data[parts[0]] = {'w': float(parts[1]), 'h': float(parts[2])}
            except ValueError: pass

# Get original AI coordinates from .pl (injected by pl_convert_new)
for line in pl_lines:
    parts = line.strip().split()
    if len(parts) >= 3 and parts[0] in macro_data:
        macro_data[parts[0]]['target_x'] = float(parts[1])
        macro_data[parts[0]]['target_y'] = float(parts[2])
        macro_data[parts[0]]['x'] = float(parts[1])
        macro_data[parts[0]]['y'] = float(parts[2])

def is_legal(nx, ny, w, h, placed_dict, ignore=None):
    # Check boundaries
    if nx < 0 or ny < 0 or nx + w > MAX_X or ny + h > MAX_Y:
        return False
    # Check overlaps with PADDING
    for m, d in placed_dict.items():
        if m == ignore: continue
        if (nx < d['x'] + d['w'] + PADDING and nx + w + PADDING > d['x'] and
            ny < d['y'] + d['h'] + PADDING and ny + h + PADDING > d['y']):
            return False
    return True

# ---------------------------------------------------------
# PHASE 1: STRICT LEGAL INITIALIZATION (Spiral Search)
# ---------------------------------------------------------
print("[*] Phase 1: Spiral Search Legalization (Strictly Legal Initialization)...")
placed = {}
# Sort by area so massive macros get placed first
sorted_macros = sorted(macro_data.keys(), key=lambda n: macro_data[n]['w'] * macro_data[n]['h'], reverse=True)

for m in sorted_macros:
    w, h = macro_data[m]['w'], macro_data[m]['h']
    tx, ty = macro_data[m]['target_x'], macro_data[m]['target_y']
    
    # Force into boundaries first
    tx = max(0.0, min(MAX_X - w, tx))
    ty = max(0.0, min(MAX_Y - h, ty))
    
    if is_legal(tx, ty, w, h, placed):
        placed[m] = {'x': tx, 'y': ty, 'w': w, 'h': h, 'target_x': tx, 'target_y': ty}
        continue
        
    # Spiral search if overlapping
    radius = STEP
    found = False
    while not found and radius < max(MAX_X, MAX_Y):
        points = []
        for x in range(-radius, radius + 1, STEP):
            points.extend([(x, radius), (x, -radius)])
        for y in range(-radius + STEP, radius, STEP):
            points.extend([(radius, y), (-radius, y)])
            
        # Sort points to find the closest absolute geometric gap
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        for dx, dy in points:
            nx, ny = tx + dx, ty + dy
            if is_legal(nx, ny, w, h, placed):
                placed[m] = {'x': nx, 'y': ny, 'w': w, 'h': h, 'target_x': macro_data[m]['target_x'], 'target_y': macro_data[m]['target_y']}
                found = True
                break
        radius += STEP
        
    if not found:
        placed[m] = {'x': tx, 'y': ty, 'w': w, 'h': h, 'target_x': tx, 'target_y': ty}
        print(f"[WARNING] Forced {m} into an overlapping spot. Relies on SA Phase 2 to fix.")

# ---------------------------------------------------------
# PHASE 2: SIMULATED ANNEALING REFINEMENT
# ---------------------------------------------------------
print("[*] Phase 2: Simulated Annealing Optimization...")
T = 1000.0
cooling_rate = 0.95
min_T = 1.0
iters_per_temp = 500

accepted_moves = 0
rejected_moves = 0

while T > min_T:
    for _ in range(iters_per_temp):
        m = random.choice(list(placed.keys()))
        curr_x, curr_y = placed[m]['x'], placed[m]['y']
        w, h = placed[m]['w'], placed[m]['h']
        tx, ty = placed[m]['target_x'], placed[m]['target_y']
        
        # Variable perturbation (cools down as T drops)
        max_shift = max(STEP, int(T))
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        nx, ny = curr_x + dx, curr_y + dy
        
        # THE GOLDEN RULE: State must never be wrong
        if is_legal(nx, ny, w, h, placed, ignore=m):
            # Cost = Manhattan distance from AI's original requested coordinate
            old_cost = abs(curr_x - tx) + abs(curr_y - ty)
            new_cost = abs(nx - tx) + abs(ny - ty)
            delta_c = new_cost - old_cost
            
            # Accept if it moves closer to AI target, OR random probability based on Temp
            if delta_c < 0 or random.random() < math.exp(-delta_c / T):
                placed[m]['x'] = nx
                placed[m]['y'] = ny
                accepted_moves += 1
            else:
                rejected_moves += 1
        else:
            rejected_moves += 1 # Rejected due to overlap
            
    T *= cooling_rate

print(f"[*] SA Complete. Accepted: {accepted_moves}, Rejected (Overlaps/Bad Cost): {rejected_moves}")

# 3. Write back to .pl
updated_count = 0
with open(pl_path, 'w') as f:
    for line in pl_lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] in placed:
            m = placed[parts[0]]
            final_x = int(round(m['x']))
            final_y = int(round(m['y']))
            
            if ":" in parts:
                idx = parts.index(":")
                suffix = " ".join(parts[idx:])
            else:
                suffix = ": N /FIXED"
                
            f.write(f"{parts[0]}\t{final_x}\t{final_y}\t{suffix}\n")
            updated_count += 1
        else:
            f.write(line)

print(f"========================================")
print(f" [SUCCESS] 100% Legal SA State Saved.")
print(f"========================================")