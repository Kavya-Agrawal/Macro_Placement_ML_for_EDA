from itertools import combinations
from heapq import *

def prim_real(vertexs_tmp, node_pos, net_info, ratio, node_info, port_info):# vertexs, edges,start='D'):
    vertexs = list(vertexs_tmp)
    if len(vertexs)<=1:
        return 0
    adjacent_dict = {}
    for node in vertexs:
        adjacent_dict[node] = []
    for node1, node2 in list(combinations(vertexs, 2)):
        if node1 in node_pos:
            rot1 = node_pos[node1][4] if len(node_pos[node1]) > 4 else False
            xo1 = net_info[node1]["x_offset"]
            yo1 = net_info[node1]["y_offset"]
            if rot1:
                hx1 = node_info[node1]["y"] / 2
                hy1 = node_info[node1]["x"] / 2
                ex1, ey1 = -yo1, xo1
            else:
                hx1 = node_info[node1]["x"] / 2
                hy1 = node_info[node1]["y"] / 2
                ex1, ey1 = xo1, yo1
            pin_x_1 = node_pos[node1][0] * ratio + hx1 + ex1
            pin_y_1 = node_pos[node1][1] * ratio + hy1 + ey1
        else:
            pin_x_1 = port_info[node1]['x']
            pin_y_1 = port_info[node1]['y']
        if node2 in node_pos:
            rot2 = node_pos[node2][4] if len(node_pos[node2]) > 4 else False
            xo2 = net_info[node2]["x_offset"]
            yo2 = net_info[node2]["y_offset"]
            if rot2:
                hx2 = node_info[node2]["y"] / 2
                hy2 = node_info[node2]["x"] / 2
                ex2, ey2 = -yo2, xo2
            else:
                hx2 = node_info[node2]["x"] / 2
                hy2 = node_info[node2]["y"] / 2
                ex2, ey2 = xo2, yo2
            pin_x_2 = node_pos[node2][0] * ratio + hx2 + ex2
            pin_y_2 = node_pos[node2][1] * ratio + hy2 + ey2
        else:
            pin_x_2 = port_info[node2]['x']
            pin_y_2 = port_info[node2]['y']
        weight = abs(pin_x_1-pin_x_2) + \
                abs(pin_y_1-pin_y_2)
        adjacent_dict[node1].append((weight, node1, node2))
        adjacent_dict[node2].append((weight, node2, node1))

    start = vertexs[0]
    minu_tree = []
    visited = set()
    visited.add(start)
    adjacent_vertexs_edges = adjacent_dict[start]
    heapify(adjacent_vertexs_edges)
    cost = 0
    cnt = 0
    while cnt < len(vertexs)-1:
        weight, v1, v2 = heappop(adjacent_vertexs_edges)
        if v2 not in visited:
            visited.add(v2)
            minu_tree.append((weight, v1, v2))
            cost += weight
            cnt += 1
            for next_edge in adjacent_dict[v2]:
                if next_edge[2] not in visited:
                    heappush(adjacent_vertexs_edges, next_edge)
    return cost
