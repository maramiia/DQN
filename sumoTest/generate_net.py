import os
import subprocess
import random
from xml.etree.ElementTree import Element, SubElement, ElementTree

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
        "--default.speed", "13.89",  # ~50 км/ч
        "--grid.attach-length", "200",  # ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: добавляет входы/выходы
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
    print("✅ Сеть 5×5 со светофорами и входами/выходами готова")

def generate_routes(train=True):
    import sumolib
    net = sumolib.net.readNet("5x5.net.xml")
    
    # 🔑 ФИКСИРУЕМ СЛУЧАЙНОСТЬ ТОЛЬКО ДЛЯ ОБУЧЕНИЯ
    if train:
        random.seed(42)  # ← ВСЕГДА ОДИНАКОВЫЙ ТРАФИК
    else:
        random.seed()    # ← ИЛИ можно вообще не вызывать — будет системный случай

    all_edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
    edge_ids = [e.getID() for e in all_edges]

    external = [eid for eid in edge_ids if "_" not in eid]
    destinations = edge_ids

    root = Element("routes")
    SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5", maxSpeed="15")

    routes = []
    for src in external or edge_ids:
        for dst in destinations:
            if src == dst:
                continue
            try:
                path = net.getOptimalPath(net.getEdge(src), net.getEdge(dst))
                if path and path[0]:
                    edges_str = " ".join(e.getID() for e in path[0])
                    routes.append((f"route_{src}_{dst}", edges_str))
            except:
                continue

    if not routes:
        raise RuntimeError("❌ Нет маршрутов!")

    # Сначала <route>
    for rid, edges_str in routes:
        SubElement(root, "route", id=rid, edges=edges_str)

    # Потом <vehicle>
    num_vehicles = 150 if train else 200  # чуть больше в тесте
    for i in range(num_vehicles):
        rid, _ = random.choice(routes)
        depart_time = i * (20 if train else random.randint(10, 30))  # в тесте — случайный интервал
        SubElement(root, "vehicle", id=f"v_{i}", type="car", route=rid, depart=str(depart_time))

    # Пики — только в тесте!
    if not train:
        peak_times = list(range(25200, 32400, random.randint(6, 12))) + list(range(61200, 68400, random.randint(6, 12)))
        for t in peak_times:
            rid, _ = random.choice(routes)
            SubElement(root, "vehicle", id=f"peak_{t}", type="car", route=rid, depart=str(t))

    filename = "train.rou.xml" if train else "test.rou.xml"
    tree = ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"✅ {filename} создан")


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
    print(f"✅ {filename} создан")


    
if __name__ == "__main__":
    for f in ["5x5.net.xml", "train.rou.xml", "test.rou.xml", "train.sumocfg", "test.sumocfg"]:
        if os.path.exists(f):
            os.remove(f)

    generate_net()  # создаёт 5x5.net.xml

    # Генерируем ДВА сценария
    generate_routes(train=True)
    generate_sumocfg("train")

    generate_routes(train=False)
    generate_sumocfg("test")

    print("\n🎉 Готово! Используйте:")
    print(" - train.sumocfg для ОБУЧЕНИЯ")
    print(" - test.sumocfg для ТЕСТИРОВАНИЯ с неопределённостью")