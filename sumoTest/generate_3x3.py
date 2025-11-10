import os
import subprocess
import random
from xml.etree.ElementTree import Element, SubElement, ElementTree

def generate_net():
    print("1. Генерация базовой сети через netgenerate...")
    # netgenerate не принимает --input-file или --tls, но создаёт сеть
    subprocess.run([
        "netgenerate",
        "--grid",
        "--grid.x-number", "3",
        "--grid.y-number", "3",
        "--grid.length", "300",
        "--default.lanenumber", "2",
        "--default.speed", "13.89",
        "-o", "3x3.net.xml"  # Краткая форма --output-file
    ], check=True)

    print("2. Добавление светофоров через netconvert...")
    # В SUMO 1.24 правильные флаги — БЕЗ --input-file и --output-file!
    subprocess.run([
        "netconvert",
        "-s", "3x3.net.xml",      # -s = --sumo-net-file (вход)
        "-o", "3x3_with_tls.net.xml",  # выход
        "--tls.guess"             # Включить автоматические светофоры
    ], check=True)

    # Заменяем оригинальный файл
    os.replace("3x3_with_tls.net.xml", "3x3.net.xml")
    print("✅ Сеть со светофорами готова: 3x3.net.xml")

def generate_routes():
    sumo_home = r"C:\Program Files (x86)\Eclipse\Sumo"
    tools = os.path.join(sumo_home, "tools")
    if tools not in os.environ.get("PYTHONPATH", ""):
        import sys
        if tools not in sys.path:
            sys.path.append(tools)

    import sumolib
    net = sumolib.net.readNet("3x3.net.xml")

    # Берём только внешние рёбра (не внутренние повороты)
    edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(":")]

    root = Element("routes")
    SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5", maxSpeed="15")

    routes = []
    for from_id in edges:
        for to_id in edges:
            if from_id != to_id:
                try:
                    path = net.getOptimalPath(net.getEdge(from_id), net.getEdge(to_id))
                    if path and path[0]:
                        edges_str = " ".join(e.getID() for e in path[0])
                        route_id = f"route_{from_id}_{to_id}"
                        routes.append((route_id, edges_str))
                except:
                    continue

    # Сначала <route>, потом <vehicle>
    for rid, edges_str in routes:
        SubElement(root, "route", id=rid, edges=edges_str)

    for i in range(120):
        rid, _ = random.choice(routes)
        SubElement(root, "vehicle", id=f"uni_{i}", type="car", route=rid, depart=str(i * 25))

    peak_times = list(range(25200, 32400, 12)) + list(range(61200, 68400, 12))
    for t in peak_times:
        rid, _ = random.choice(routes)
        SubElement(root, "vehicle", id=f"peak_{t}", type="car", route=rid, depart=str(t))

    tree = ElementTree(root)
    tree.write("3x3.rou.xml", encoding="utf-8", xml_declaration=True)
    print("✅ Маршруты созданы: 3x3.rou.xml")

def generate_sumocfg():
    with open("3x3.sumocfg", "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="3x3.net.xml"/>
        <route-files value="3x3.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="7200"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>""")
    print("✅ Конфигурация создана: 3x3.sumocfg")

if __name__ == "__main__":
    # Удаляем старые файлы (опционально)
    for f in ["3x3.net.xml", "3x3.rou.xml", "3x3.sumocfg"]:
        if os.path.exists(f):
            os.remove(f)

    generate_net()
    generate_routes()
    generate_sumocfg()
    print("\n🎉 ВСЁ ГОТОВО! Запускайте: sumo-gui -c 3x3.sumocfg")