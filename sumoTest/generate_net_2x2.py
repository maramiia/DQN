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
    print("sumolib не найден")
    sys.exit(1)

def generate_net():
    print("1. Генерация сети: два перекрёстка...")
    subprocess.run([
        "netgenerate",
        "--grid", "--grid.x-number", "2", "--grid.y-number", "1",
        "--grid.x-length", "400", "--grid.y-length", "300",
        "--default.lanenumber", "3", "--default.speed", "13.89",
        "--grid.attach-length", "250",
        "--output", "dual_crossings.net.xml"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "netconvert",
        "--sumo-net-file", "dual_crossings.net.xml",
        "--output", "dual_crossings.net.xml",
        "--tls.guess", "--tls.guess.threshold", "1",
        "--no-turnarounds", "true"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Сеть готова")

def generate_routes():
    print("2. Генерация маршрутов с ВЕРТИКАЛЬНЫМ трафиком...")
    net = sumolib.net.readNet("dual_crossings.net.xml")
    root = Element("routes")
    SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.2", length="5", maxSpeed="15")

    vehicles = []

    for zone, node_id in [("baseline", "A0"), ("drqn", "B0")]:
        inputs = []
        outputs = []
        for edge in net.getEdges():
            if ":" in edge.getID():
                continue
            from_n = edge.getFromNode().getID()
            to_n = edge.getToNode().getID()
            if to_n == node_id and from_n != node_id:
                inputs.append(edge.getID())
            elif from_n == node_id and to_n != node_id:
                outputs.append(edge.getID())

        print(f"  Зона '{zone}': входы={inputs}, выходы={outputs}")

        routes = []
        for src in inputs:
            for dst in outputs:
                if src == dst:
                    continue
                try:
                    path = net.getShortestPath(net.getEdge(src), net.getEdge(dst))
                    if path and path[0]:
                        rid = f"r_{zone}_{len(routes)}"
                        edges_str = " ".join(e.getID() for e in path[0])
                        SubElement(root, "route", id=rid, edges=edges_str)
                        routes.append(rid)
                except:
                    continue

        for i in range(500):
            t = random.randint(0, 1999)
            if routes:
                rid = random.choice(routes)
                vehicles.append((t, f"v_{zone}_{i}", rid))

    vehicles.sort(key=lambda x: x[0])
    for t, vid, rid in vehicles:
        SubElement(root, "vehicle", id=vid, type="car", route=rid, depart=str(t))

    ElementTree(root).write("dual_heavy.rou.xml", encoding="utf-8", xml_declaration=True)
    print(f"Маршруты готовы: {len(vehicles)} машин (2500 на зону)")

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
        <end value="2000"/> <!-- 2000 секунд -->
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>""")
    print("Конфиг создан (2000 сек)")

if __name__ == "__main__":
    for f in ["dual_crossings.net.xml", "dual_heavy.rou.xml", "dual_heavy.sumocfg"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑 Удалён {f}")
    generate_net()
    generate_routes()
    generate_sumocfg()
    print("\nГотово!")