# generate_heavy_test_routes.py
import os
import random
import sumolib
from xml.etree.ElementTree import Element, SubElement, ElementTree

def generate_heavy_test_routes():
    print("🚛 Генерация сценария с высокой плотностью трафика (heavy_test)...")
    
    # Проверка наличия сети
    if not os.path.exists("5x5.net.xml"):
        raise FileNotFoundError("❌ Не найден 5x5.net.xml. Запустите generate_sumo.py сначала.")
    
    net = sumolib.net.readNet("5x5.net.xml")
    random.seed()  # системная случайность → реалистичные пики

    # Получаем все рёбра и отделяем внешние (источники/стоки)
    all_edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
    edge_ids = [e.getID() for e in all_edges]
    external = [eid for eid in edge_ids if "_" not in eid]
    destinations = edge_ids

    if not external:
        raise RuntimeError("❌ Не найдено внешних рёбер (например: left0, top2). Убедитесь, что сеть содержит входы.")

    # === Создание XML-структуры ===
    root = Element("routes")
    SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5", maxSpeed="15")

    # Генерация маршрутов (src → dst)
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
            except Exception as e:
                continue  # пропускаем недостижимые пары

    if not routes:
        raise RuntimeError("❌ Не удалось сгенерировать ни одного маршрута.")

    # Сохраняем <route>
    for rid, edges_str in routes:
        SubElement(root, "route", id=rid, edges=edges_str)

    # === Генерация машин ===
    vehicles = []
    srcs = list(routes_by_src.keys())

    # 🔥 1. Базовый поток: 600 машин за 1 час (≈0.17 машин/сек → интервал 2–6 сек)
    print("  → Генерация базового потока (600 машин)...")
    current_time = 0
    for i in range(600):
        src = srcs[i % len(srcs)]  # round-robin балансировка
        if src in routes_by_src and routes_by_src[src]:
            rid, _ = random.choice(routes_by_src[src])
        else:
            rid, _ = random.choice(routes)
        # интервал: 2–6 секунд
        interval = random.randint(2, 6)
        current_time += interval
        vehicles.append((current_time, f"v_{i}", rid))

    # 🔥 2. Интенсивные пики: утро (25200–32400) и вечер (61200–68400)
    # Цель: плотность ~1 машина / 1–2 сек → ~3600 / 1.5 ≈ 2400 машин за 1 час!
    # Но SUMO не выдержит 2400 → ограничимся по 300 в каждый пик (~1 машина / 2–3 сек)
    print("  → Генерация пиковых потоков (по 300 машин в утро и вечер)...")
    def add_peak(start_sec, count):
        times = []
        t = start_sec
        for _ in range(count):
            times.append(t)
            t += random.randint(1, 3)  # плотность: 1–3 сек
        return times

    peak_times = add_peak(25200, 300) + add_peak(61200, 300)

    for t in peak_times:
        src = random.choice(srcs)
        if src in routes_by_src and routes_by_src[src]:
            rid, _ = random.choice(routes_by_src[src])
        else:
            rid, _ = random.choice(routes)
        vehicles.append((t, f"peak_{len(vehicles)}", rid))

    # 🔥 3. Дополнительно: «всплеск» в начале симуляции (первые 5 минут)
    # Для быстрой загрузки и проверки в traffic_analyzer
    print("  → Добавление всплеска в начало симуляции (50 машин за первые 120 сек)...")
    t = 0
    for i in range(50):
        src = random.choice(srcs)
        if src in routes_by_src and routes_by_src[src]:
            rid, _ = random.choice(routes_by_src[src])
        else:
            rid, _ = random.choice(routes)
        t += random.randint(2, 4)  # 2–4 сек интервал
        vehicles.append((t, f"burst_{i}", rid))

    # Сортируем по времени отправления (обязательно!)
    vehicles.sort(key=lambda x: x[0])

    # Создаём <vehicle>
    for depart_time, vid, rid in vehicles:
        SubElement(root, "vehicle", id=vid, type="car", route=rid, depart=str(depart_time))

    # Сохраняем
    filename = "heavy_test.rou.xml"
    tree = ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"✅ {filename} создан (всего машин: {len(vehicles)})")

def generate_heavy_test_sumocfg():
    print("⚙️  Генерация heavy_test.sumocfg...")
    with open("heavy_test.sumocfg", "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="5x5.net.xml"/>
        <route-files value="heavy_test.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="7200"/> <!-- 2 часа -->
    </time>
    <processing>
        <time-to-teleport value="-1"/> <!-- машины не исчезают, а ждут -->
    </processing>
</configuration>""")
    print("✅ heavy_test.sumocfg создан")

if __name__ == "__main__":
    # Удаляем старые тяжёлые файлы, если есть
    for f in ["heavy_test.rou.xml", "heavy_test.sumocfg"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"🗑 Старый {f} удалён.")

    try:
        generate_heavy_test_routes()
        generate_heavy_test_sumocfg()
        print("\n🎉 Готово! Для запуска:")
        print("   - Скопируйте `heavy_test.sumocfg` в папку с SUMO-проектом")
        print("   - Или используйте в `test_drqn.py`: sumo_cfg='heavy_test.sumocfg'")
    except Exception as e:
        print(f"❌ Ошибка: {e}")