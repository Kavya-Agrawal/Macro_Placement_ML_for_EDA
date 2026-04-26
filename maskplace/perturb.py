#!/usr/bin/env python3
import sys
import os

if len(sys.argv) != 3:
    print("Usage: ./pl_convert_new <friend_filename> <foldername>")
    sys.exit(1)

friend_file = sys.argv[1]
folder_name = sys.argv[2]
base_dir = f"/home/anubhav/mleda/test_ispd2005/{folder_name}"

friend_path = os.path.join(base_dir, friend_file)
nodes_path = os.path.join(base_dir, f"{folder_name}.nodes")
target_pl_path = os.path.join(base_dir, f"{folder_name}.pl")

# ==========================================
# TOGGLE THIS IF MACROS STILL OVERLAP
# True = Assumes AI outputs the Center (Subtracts W/2, H/2)
# False = Assumes AI outputs Bottom-Left (Just rounds to Int)
SHIFT_FROM_CENTER = True 
# ==========================================

# 1. Parse the .nodes file to get Width and Height for every macro
node_dims = {}
with open(nodes_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        # Node lines look like: o210904 100 200
        if len(parts) >= 3 and parts[0].startswith('o'):
            try:
                w = float(parts[1])
                h = float(parts[2])
                node_dims[parts[0]] = (w, h)
            except ValueError:
                pass

# 2. Parse the friend's AI output file
new_macros = {}
with open(friend_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            node_name = parts[0]
            raw_x = float(parts[1])
            raw_y = float(parts[2])
            
            if SHIFT_FROM_CENTER and node_name in node_dims:
                w, h = node_dims[node_name]
                final_x = int(round(raw_x - (w / 2.0)))
                final_y = int(round(raw_y - (h / 2.0)))
            else:
                final_x = int(round(raw_x))
                final_y = int(round(raw_y))
                
            new_macros[node_name] = (final_x, final_y)

# 3. Safely update the .pl file
updated_lines = []
update_count = 0

with open(target_pl_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 0 and parts[0] in new_macros:
            new_x, new_y = new_macros[parts[0]]
            
            if ":" in parts:
                idx = parts.index(":")
                suffix = " ".join(parts[idx:])
            else:
                suffix = ": N"
                
            updated_lines.append(f"{parts[0]}\t{new_x}\t{new_y}\t{suffix}\n")
            update_count += 1
        else:
            updated_lines.append(line)

# 4. Overwrite original .pl
with open(target_pl_path, 'w') as f:
    f.writelines(updated_lines)

print(f"========================================")
print(f" [SUCCESS] NEW Macro Conversion Complete")
print(f" Mode: {'CENTER -> BOTTOM-LEFT' if SHIFT_FROM_CENTER else 'BOTTOM-LEFT (INT ROUNDING)'}")
print(f" Inserted {update_count} macro coordinates safely.")
print(f"========================================")