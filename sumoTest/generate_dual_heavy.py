# generate_dual_heavy.py (исправлено)
import os
import subprocess
import random
from xml.etree.ElementTree import Element, SubElement, ElementTree
import sys

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

try:
    import sumolib
except ImportError:
    print("❌ sumolib не найден")
    sys.exit(1)

def generate_net():
    print("1. Генерация сети: два перекрёстка...")
    subprocess.run([
        "netgenerate",
        "--grid",
        "--grid.x-number", "2",
        "--grid.y-number", "1",
        "--grid.x-length", "400",
        "--grid.y-length", "300",
        "--default.lanenumber", "3",
        "--default.speed", "13.89",
        "--grid.attach-length", "200",
        "--output", "dual_crossings.net.xml"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "netconvert",
        "--sumo-net-file", "dual_crossings.net.xml",
        "--output", "dual_crossings.net.xml",
        "--tls.guess",
        "--tls.guess.threshold", "10",
        "--no-turnarounds", "true"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Сеть готова")

def generate_routes():
    print("2. Генерация маршрутов...")
    net = sumolib.net.readNet("dual_crossings.net.xml")
    all_edges = [e for e in net.getEdges() if ":" not in e.getID()]
    edge_ids = [e.getID() for e in all_edges]

    # Внешние рёбра (входы/выходы) — не содержат '/'
    external = [eid for eid in edge_ids if "/" not in eid]
    print(f"  Внешние рёбра: {external}")

    if len(external) < 4:
        raise RuntimeError("Недостаточно внешних рёбер! Нужны как минимум 4 (2 слева, 2 справа).")

    # Разделим вручную: левые и правые
    left_sources = [e for e in external if "left" in e]
    right_sources = [e for e in external if "right" in e]

    if not left_sources or not right_sources:
        # Fallback: первые два — левые, последние два — правые
        left_sources = external[:2]
        right_sources = external[2:]

    print(f"  Левые входы: {left_sources}")
    print(f"  Правые входы: {right_sources}")

    # Все рёбра для каждой зоны — просто по координатам узлов
    def get_zone_edges(zone_x):
        edges = []
        for edge in all_edges:
            from_node = net.getNode(edge.getFromNode().getID())
            to_node = net.getNode(edge.getToNode().getID())
            # Если хотя бы один узел в нужной колонке — относим к зоне
            if from_node.getCoord()[0] // 400 == zone_x or to_node.getCoord()[0] // 400 == zone_x:
                edges.append(edge.getID())
        return edges

    left_edges = get_zone_edges(0)   # x=0
    right_edges = get_zone_edges(1)  # x=1

    root = Element("routes")
    SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.2", length="5", maxSpeed="15")

    vehicles = []
    depart_time = 0

    for zone_name, sources, dests in [("baseline", left_sources, left_edges), ("drqn", right_sources, right_edges)]:
        route_count = 0
        for src in sources:
            for dst in dests:
                if src == dst:
                    continue
                try:
                    path = net.getShortestPath(net.getEdge(src), net.getEdge(dst))
                    if path and path[0]:
                        edges_str = " ".join(e.getID() for e in path[0])
                        rid = f"route_{zone_name}_{route_count}"
                        SubElement(root, "route", id=rid, edges=edges_str)

                        # Генерируем 30 машин на маршрут
                        for i in range(30):
                            vehicles.append((depart_time, f"v_{zone_name}_{route_count}_{i}", rid))
                            depart_time += random.randint(4, 10)
                        route_count += 1
                except Exception as ex:
                    continue
        print(f"  Зона '{zone_name}': {route_count} маршрутов")

    vehicles.sort(key=lambda x: x[0])
    for depart_time, vid, rid in vehicles:
        SubElement(root, "vehicle", id=vid, type="car", route=rid, depart=str(depart_time))

    tree = ElementTree(root)
    tree.write("dual_heavy.rou.xml", encoding="utf-8", xml_declaration=True)
    print(f"✅ Маршруты готовы: {len(vehicles)} машин")

def generate_sumocfg():
    with open("dual_heavy.sumocfg", "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="dual_crossings.net.xml"/>
        <route-files value="dual_heavy.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>""")
    print("✅ Конфиг создан")

if __name__ == "__main__":
    for f in ["dual_crossings.net.xml", "dual_heavy.rou.xml", "dual_heavy.sumocfg"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑 Удалён {f}")
    generate_net()
    generate_routes()
    generate_sumocfg()
    print("\n🎉 Готово!")