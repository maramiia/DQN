import os
import subprocess
import random
from xml.etree.ElementTree import Element, SubElement, ElementTree
import sumolib


def generate_net():
    print("1. Генерация расширенной сети с входами/выходами...")
    subprocess.run([
        "netgenerate",
        "--grid",
        "--grid.x-number", "5",
        "--grid.y-number", "5",
        "--grid.x-length", "400",
        "--grid.y-length", "350",
        "--default.lanenumber", "3",
        "--default.speed", "13.89",
        "--grid.attach-length", "200",
        "-o", "5x5.net.xml"
    ], check=True)

    print("2. Добавление светофоров...")
    subprocess.run([
        "netconvert",
        "-s", "5x5.net.xml",
        "-o", "5x5_with_tls.net.xml",
        "--tls.guess",
        "--tls.guess.threshold", "10",
        "--no-turnarounds", "true"
    ], check=True)

    os.replace("5x5_with_tls.net.xml", "5x5.net.xml")
    print("Сеть 5×5 со светофорами и входами/выходами готова")


def generate_routes(train=True):
    net = sumolib.net.readNet("5x5.net.xml")
    
    if train:
        random.seed(42)
    else:
        random.seed() 

    all_edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
    edge_ids = [e.getID() for e in all_edges]

    external = [eid for eid in edge_ids if "_" not in eid]
    destinations = edge_ids

    root = Element("routes")
    SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5", maxSpeed="15")

    routes = []
    routes_by_src = {}

    for src in external:
        routes_by_src[src] = []
        for dst in destinations:
            if src == dst:
                continue
            try:
                path = net.getOptimalPath(net.getEdge(src), net.getEdge(dst))
                if path and path[0]:
                    edges_str = " ".join(e.getID() for e in path[0])
                    rid = f"route_{src}_{dst}"
                    routes.append((rid, edges_str))
                    routes_by_src[src].append((rid, edges_str))
            except Exception:
                continue

    if not routes:
        raise RuntimeError("Нет маршрутов!")

    for rid, edges_str in routes:
        SubElement(root, "route", id=rid, edges=edges_str)

    vehicles = []
    num_vehicles = 150 if train else 200

    srcs = list(routes_by_src.keys())
    if not srcs:
        srcs = [r[0].split('_')[1] for r in routes]

    for i in range(num_vehicles):
        src = srcs[i % len(srcs)] 
        if src in routes_by_src and routes_by_src[src]:
            rid, _ = random.choice(routes_by_src[src])
        else:
            rid, _ = random.choice(routes)
        
        if i == 0:
            depart_time = 0
        else:
            interval = 20 if train else random.randint(10, 30)
            depart_time = vehicles[-1][0] + interval
        
        vehicles.append((depart_time, f"v_{i}", rid))


    if not train:
        peak_times = []
        t = 25200
        while t < 32400:
            peak_times.append(t)
            t += random.randint(6, 12)
        t = 61200
        while t < 68400:
            peak_times.append(t)
            t += random.randint(6, 12)
        
        for t in peak_times:
            src = random.choice(srcs)
            if src in routes_by_src and routes_by_src[src]:
                rid, _ = random.choice(routes_by_src[src])
            else:
                rid, _ = random.choice(routes)
            vehicles.append((t, f"peak_{t}", rid))

    vehicles.sort(key=lambda x: x[0])

    for depart_time, vid, rid in vehicles:
        SubElement(root, "vehicle", id=vid, type="car", route=rid, depart=str(depart_time))

    filename = "train.rou.xml" if train else "test.rou.xml"
    tree = ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"{filename} создан (машин: {len(vehicles)})")


def generate_sumocfg(name="train"):
    filename = f"{name}.sumocfg"
    net_file = "5x5.net.xml"
    route_file = f"{name}.rou.xml"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="7200"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>""")
    print(f"{filename} создан")


if __name__ == "__main__":
    for f in ["5x5.net.xml", "train.rou.xml", "test.rou.xml", "train.sumocfg", "test.sumocfg"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑 Удалён {f}")

    generate_net()

    generate_routes(train=True)
    generate_sumocfg("train")

    generate_routes(train=False)
    generate_sumocfg("test")

    print("\nГотово! Используйте:")
    print(" - train.sumocfg для ОБУЧЕНИЯ (150 машин, сбалансировано, сортировано)")
    print(" - test.sumocfg для ТЕСТИРОВАНИЯ (200+ машин + пики, сортировано)")
