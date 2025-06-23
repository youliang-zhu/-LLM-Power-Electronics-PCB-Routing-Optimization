import heapq
import numpy as np
import requests
import json
import math
import re
from typing import List, Tuple, Dict, Optional
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import time  

from library.llm_optimized_Astar import *
from library.get_routing_order import *

def main():
    print("开始PCB布线测试...")

    start_time = time.time()

    with open(r'E://leuven//papers//ECCEUK2025//LLM_Astar//code-deepseek//layout.json', 'r') as f:
        layout_json = f.read()
    
    router = PCBRouter(300)  
    wires = get_routing_order(layout_json)
    print("布线顺序为：", wires)
    
    if router.route_all_wires(wires):
        print("\n所有线路布线成功！")
        # 打印路径信息
        # for i, path in enumerate(router.paths):
        #     print(f"\n线路{i+1}的路径信息：")
        #     print(f"  起点: {path[0]}")
        #     print(f"  终点: {path[-1]}")
        #     print(f"  路径长度: {len(path)}个单位")
        #     print(f"  转折点: {[p for i, p in enumerate(path) if i > 0 and i < len(path)-1 and (p[0] != path[i-1][0] or p[1] != path[i+1][1])]}")
    else:
        print("\n布线失败！")

    end_time = time.time() 
    elapsed_time = end_time - start_time  
    print(f"\n程序运行时间: {elapsed_time:.2f}秒")


if __name__ == "__main__":
    main()