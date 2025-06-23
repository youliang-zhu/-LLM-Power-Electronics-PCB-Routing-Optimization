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

from library.llm_api import *


def get_routing_order(layout_json):
    # 解析JSON数据
    layout_data = json.loads(layout_json)
    networks = layout_data["networks"]
    components = layout_data["components"]
    
    # 提取所有线段信息
    segments = []
    for net_name, net_info in networks.items():
        path = net_info["path"]
        # 将每条线拆分成线段
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            segment = {
                "net_name": net_name,
                "segment_id": f"{net_name}_seg{i}",
                "start_pad": start["pad"],
                "end_pad": end["pad"],
                "start_coord": start["coordinates"],
                "end_coord": end["coordinates"],
                # 计算线段长度
                "length": math.sqrt(
                    (end["coordinates"][0] - start["coordinates"][0])**2 + 
                    (end["coordinates"][1] - start["coordinates"][1])**2
                )
            }
            segments.append(segment)
    
    # 针对deepseek-coder优化的提示词，与其响应模式相匹配
    prompt = """作为PCB布线专家，请为以下PCB线段提供最佳布线顺序。

线段信息:
"""
    
    # 添加所有线段的信息
    for seg in segments:
        prompt += f"{seg['segment_id']}:\n"
        prompt += f"- 网络: {seg['net_name']}\n"
        prompt += f"- 从: ({seg['start_coord'][0]}, {seg['start_coord'][1]})\n"
        prompt += f"- 到: ({seg['end_coord'][0]}, {seg['end_coord'][1]})\n"
        prompt += f"- 长度: {seg['length']:.2f}\n\n"

    prompt += """


【布线顺序优化核心原则】:
1. **识别关键路径和瓶颈**:
   - 分析哪些线段位于高密度区域或关键通道
   - 识别如果后布线将导致严重绕道的线段

2. **考虑线段间相互影响**:
   - 评估每个线段的布线对后续线段可用空间的影响
   - 避免过早布线会形成"障碍墙"的线段

3. **灵活的网络交错布线**:
   - 不要按网络名称连续布线（例如，不要一次性布完所有GND线段）
   - 根据空间布局和线段位置灵活交错不同网络的布线顺序

4. **优先级评估因素**:
   - 线段长度：较短线段通常灵活性更高，可后布线
   - 位置关键性：某些位置只有有限的布线选择，应优先处理


请直接给出所有线段ID的推荐布线顺序。

推荐布线顺序:
"""

    # 调用API
    api = DEEPSEEKAPI()
    response = api.get_llm_response(prompt)
    
    print("\n大模型的布线分析结果：")
    print(response)
    
    # 提取线段ID
    extracted_ids = []
    
    # 分离分析和线段ID部分
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 尝试匹配线段ID - deepseek-coder模型通常会以纯文本返回ID
        for seg in segments:
            if seg['segment_id'] in line and seg['segment_id'] not in extracted_ids:
                extracted_ids.append(seg['segment_id'])
                break
    
    if extracted_ids:
        print("\n推荐的线段布线顺序：")
        for seg_id in extracted_ids:
            print(seg_id)
        
        # 构建有序线段列表
        ordered_segments = []
        processed_segments = set()
        
        for seg_id in extracted_ids:
            matching_segment = next(
                (seg for seg in segments if seg['segment_id'] == seg_id),
                None
            )
            if matching_segment:
                ordered_segments.append([
                    matching_segment['start_coord'],
                    matching_segment['end_coord']
                ])
                processed_segments.add(seg_id)
        
        # 添加遗漏的线段
        for segment in segments:
            if segment['segment_id'] not in processed_segments:
                ordered_segments.append([
                    segment['start_coord'],
                    segment['end_coord']
                ])
                print(f"提示：线段 {segment['segment_id']} 添加到布线末尾")
    else:
        # 未能提取任何线段ID，使用简单的启发式方法
        print("未能从模型响应中提取线段顺序，使用启发式排序...")
        
        # 按网络类型和长度排序
        sorted_segments = sorted(
            segments,
            key=lambda x: (
                0 if x['net_name'] in ["GND", "12V"] else 1,
                x['length']
            )
        )
        
        ordered_segments = [[seg['start_coord'], seg['end_coord']] for seg in sorted_segments]
    
    return ordered_segments