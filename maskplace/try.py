from place_db import PlaceDB

placedb = PlaceDB("adaptec1")
heuristic = placedb.node_id_to_name.copy()

print(heuristic)