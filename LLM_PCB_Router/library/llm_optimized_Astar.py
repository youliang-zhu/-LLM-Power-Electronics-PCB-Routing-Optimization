import heapq
import numpy as np
import json
import math
import re
from typing import List, Tuple, Dict, Optional
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

from library.llm_api import *

class PCBRouter:
    def __init__(self, size: int = 10):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.paths = []
        self.colors = []
        self.api = DEEPSEEKAPI()
        self.weight_radius = 3  # 增加权重影响半径
        self.cross_points = []
        self.recommended_turns = []
        self.occupied_weights = {}
        self.endpoints = set()
        self.weight_map = {}

    def format_path(self, path: List[Tuple[int, int]]) -> str:
        if not path:
            return ""
        return f"{path[0]}->{path[-1]}"
        
    def generate_llm_prompt(self, start: Tuple[int, int], end: Tuple[int, int]) -> str:
        """
        生成LLM提示词，支持更智能的动态权重分配
        """
        paths_str = "; ".join([self.format_path(path) for path in self.paths])
        
        prompt = f"""假设你是一个资深的pcb工程师，请你为{self.size}x{self.size}网格PCB布线分析路径规划。

        任务：为从{start}到{end}的布线提供最优避障方案。
        已有布线路径：[{paths_str}]

        【核心要求】：
        1. 必须完全避免与现有线路相交，这是最高优先级要求
        2. 应提供足够远离现有线路的路径建议
        3. 路径可以很长，但绝对不能有交叉
        4. 优先使用直角(90度)转弯，其次是45度转弯
        5. 尽量减少转弯次数，保持路径美观简洁

        【布线风格要求】：
        1. 优先选择直线段路径，避免不必要的zigzag
        2. 在不影响功能的前提下，追求美观布局
        3. 路径应保持至少2-3个单位的间距，除非空间限制

        请仅返回如下格式的JSON（保持键名双引号）：
        {{
            "cross_points": [
                {{
                    "x": 3,
                    "y": 2,
                    "weight": 10.0  // 绝对禁止通过的区域必须使用最高权重20.0
                }}
            ],
            "recommended_turns": [
                {{
                    "x": 4,
                    "y": 2,
                    "reason": "避开已有线路交叉点"
                }}
            ]
        }}

        权重指南：
        - 20.0：绝对禁止通过的区域（如现有线路及其周围1-2格范围）
        - 8.0-10.0：高风险区域，与现有线路接近，极可能导致交叉
        - 5.0-7.9：中风险区域，可能导致布线困难
        - 1.0-4.9：低风险区域

        请至少提供5-8个潜在的交叉风险点，并确保权重分配合理
        请至少提供2-3个推荐转折点，这些点应帮助线路完全避开已有布线
        """

        return prompt

    def parse_llm_response(self, response: str) -> Tuple[List[Dict], List[Dict]]:
        try:
            response = response.strip()
            response = re.sub(r'//.*$', '', response, flags=re.MULTILINE)  # 去除单行注释
            response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
            
            # 先打印原始响应，用于调试
            print("原始LLM响应:", response)
            
            # 尝试提取 JSON 部分
            json_start = response.find('{')
            json_end = response.rfind('}')
            if json_start != -1 and json_end != -1:
                response = response[json_start:json_end + 1]
            
            # 清理所有可能的格式标记
            for marker in ['```json', '```javascript', '```']:
                if marker in response:
                    parts = response.split(marker)
                    for part in parts:
                        if '{' in part and '}' in part:
                            response = part[part.find('{'):part.rfind('}')+1]
                            break
            
            # 基本清理
            response = response.strip()
            
            # 如果响应为空，返回默认值
            if not response or not response.startswith('{') or not response.endswith('}'):
                print("警告：未找到有效的JSON结构，使用默认值")
                return [], []
            
            try:
                # 第一次尝试：直接解析
                data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"第一次JSON解析失败，尝试修复响应... 错误: {e}")
                try:
                    # 更强大的JSON修复逻辑
                    # 1. 规范化引号
                    response = response.replace("'", '"')
                    # 2. 修复键名引号
                    response = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', response)
                    # 3. 修复值中的引号问题
                    response = re.sub(r':"([^"]*)"', r':"\1"', response)
                    # 4. 清理多余的逗号
                    response = re.sub(r',\s*([}\]])', r'\1', response)
                    # 5. 修复数值格式
                    response = re.sub(r':\s*"(\d+\.?\d*)"', r':\1', response)
                    
                    print("修复后的JSON:", response)
                    data = json.loads(response)
                except json.JSONDecodeError as e2:
                    print(f"JSON修复失败，返回默认值。错误: {e2}")
                    print(f"最终尝试解析的字符串: {response}")
                    return [], []
            
            # 提取和验证数据
            cross_points = []
            recommended_turns = []
            
            # 验证并提取交叉点
            if "cross_points" in data and isinstance(data["cross_points"], list):
                for point in data["cross_points"]:
                    if isinstance(point, dict) and all(k in point for k in ['x', 'y', 'weight']):
                        try:
                            validated_point = {
                                'x': int(point['x']),
                                'y': int(point['y']),
                                'weight': float(point['weight'])
                            }
                            cross_points.append(validated_point)
                        except (ValueError, TypeError):
                            continue
            
            # 验证并提取推荐转折点
            if "recommended_turns" in data and isinstance(data["recommended_turns"], list):
                for turn in data["recommended_turns"]:
                    if isinstance(turn, dict) and all(k in turn for k in ['x', 'y']):
                        try:
                            validated_turn = {
                                'x': int(turn['x']),
                                'y': int(turn['y'])
                            }
                            if 'reason' in turn:
                                validated_turn['reason'] = str(turn['reason'])
                            recommended_turns.append(validated_turn)
                        except (ValueError, TypeError):
                            continue
            
            self.cross_points.extend(cross_points)
            self.recommended_turns.extend(recommended_turns)
            
            return cross_points, recommended_turns
            
        except Exception as e:
            print(f"解析LLM响应失败，使用默认值。错误: {e}")
            return [], []


    def calculate_weight_map(self, cross_points: List[Dict], recommended_turns: List[Dict]) -> Dict[Tuple[int, int], float]:
        weights = self.occupied_weights.copy()
        
        # 显著提高已占用点的权重
        for pos in self.occupied_weights:
            if pos not in self.endpoints:  # 端点保持较低权重
                weights[pos] *= 8.0  # 大幅提高已占用点的权重
        
        # 在已占用点周围创建更大的高权重区域
        for pos in list(self.occupied_weights.keys()):
            if pos not in self.endpoints:  # 不处理端点
                x, y = pos
                for dx in range(-self.weight_radius, self.weight_radius + 1):
                    for dy in range(-self.weight_radius, self.weight_radius + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.size and 0 <= ny < self.size:
                            new_pos = (nx, ny)
                            if new_pos not in self.endpoints:
                                distance = math.sqrt(dx*dx + dy*dy)
                                # 使用指数衰减函数创建更平滑的权重分布
                                weight = 5.0 * math.exp(-distance/2.0)
                                weights[new_pos] = max(weights.get(new_pos, 0), weight)
        
        # 处理交叉点，显著提高权重以强制绕行
        for point in cross_points:
            x, y = point["x"], point["y"]
            base_weight = float(point["weight"]) * 2  # 加倍交叉点权重
            
            for dx in range(-self.weight_radius, self.weight_radius + 1):
                for dy in range(-self.weight_radius, self.weight_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        new_pos = (nx, ny)
                        if new_pos not in self.endpoints:
                            distance = math.sqrt(dx*dx + dy*dy)
                            weight = base_weight * math.exp(-distance/1.5)
                            weights[new_pos] = max(weights.get(new_pos, 0), weight)
        
        # 处理推荐转折点
        for point in recommended_turns:
            x, y = point["x"], point["y"]
            if 0 <= x < self.size and 0 <= y < self.size:
                weights[(x, y)] = min(weights.get((x, y), 1.0), 0.3)  # 降低推荐转折点的权重
        
        return weights

    def get_movement_cost(self, current: Tuple[int, int], next_pos: Tuple[int, int], 
                         prev_pos: Optional[Tuple[int, int]] = None) -> float:
        """改进的移动代价计算，增加转弯惩罚"""
        base_cost = math.sqrt((current[0] - next_pos[0])**2 + (current[1] - next_pos[1])**2)
        
        if prev_pos:
            prev_direction = (current[0] - prev_pos[0], current[1] - prev_pos[1])
            next_direction = (next_pos[0] - current[0], next_pos[1] - current[1])
            
            if prev_direction != next_direction:
                # 显著增加转弯惩罚
                if abs(prev_direction[0]) != abs(next_direction[0]) or \
                   abs(prev_direction[1]) != abs(next_direction[1]):
                    base_cost *= 2.0  # 90度转弯惩罚加大
                else:
                    base_cost *= 1.5  # 45度转弯惩罚加大
                
                # 检查是否在推荐转折点附近
                for point in self.recommended_turns:
                    if abs(next_pos[0] - point["x"]) <= 1 and abs(next_pos[1] - point["y"]) <= 1:
                        base_cost *= 0.5  # 在推荐转折点附近降低转弯代价
        
        return base_cost

    def improved_heuristic(self, current: Tuple[int, int], end: Tuple[int, int]) -> float:
        """改进的启发式函数，考虑障碍物和交叉点"""
        # 基础距离估计
        manhattan_dist = abs(current[0] - end[0]) + abs(current[1] - end[1])
        euclidean_dist = math.sqrt((current[0] - end[0])**2 + (current[1] - end[1])**2)
        
        # 综合两种距离度量
        base_distance = (manhattan_dist + euclidean_dist) / 2
        
        # 考虑路径上的障碍物
        obstacles_penalty = 0
        line_points = self.get_line_points(current, end)
        for point in line_points:
            if point in self.occupied_weights and point not in self.endpoints:
                obstacles_penalty += 2.0
        
        # 考虑交叉点的影响
        for point in self.cross_points:
            x, y = point["x"], point["y"]
            if (x, y) in line_points:
                obstacles_penalty += point["weight"]
        
        return base_distance + obstacles_penalty
    
    def get_line_points(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取两点之间直线上的所有格点"""
        points = []
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append((x1, y1))
        return points

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居节点，允许经过端点"""
        x, y = pos
        neighbors = []
        # 八个方向：上下左右 + 四个对角线
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                # 如果是端点或未被占用，则允许通过
                if new_pos in self.endpoints or self.grid[new_x][new_y] == 0:
                    neighbors.append(new_pos)
        return neighbors

    def a_star_with_llm(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        prompt = self.generate_llm_prompt(start, end)
        response = self.api.get_llm_response(prompt)
        cross_points, recommended_turns = self.parse_llm_response(response)

        # print("\n大模型响应输出为:\n", response)
        
        self.weight_map = self.calculate_weight_map(cross_points, recommended_turns)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.improved_heuristic(start, end)}
        open_set_hash = {start}
        prev_points = {start: None}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                movement_cost = self.get_movement_cost(current, neighbor, prev_points.get(current))
                weight = self.weight_map.get(neighbor, 1.0)
                tentative_g = g_score[current] + movement_cost * weight
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    prev_points[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.improved_heuristic(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return None
    
    def visualize_routing(self, wires: List[List[Tuple[int, int]]]):
        """可视化布线结果"""
        # 创建图形，调整图形大小和边距
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(right=0.82)  # 调整右边距，为两列图例留出空间
        ax = plt.gca()
        
        # 绘制网格底色
        ax.add_patch(Rectangle((0, 0), self.size, self.size, 
                            facecolor='lightgray', alpha=0.3))
        
        # 设置固定间隔的刻度
        step = 25
        xticks = np.arange(0, self.size + step, step)
        yticks = np.arange(0, self.size + step, step)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
        # 绘制主网格线
        ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
        
        # 为每条完整线路分配一个颜色
        n_wires = len(wires)
        colors = plt.cm.tab20(np.linspace(0, 1, n_wires))  # 使用tab20色板，提供更多颜色选择
        
        # 按原始线路绘制路径
        legend_elements = []
        
        # 直接遍历self.paths，每个path对应一个wire
        for wire_idx, (wire, path) in enumerate(zip(wires, self.paths)):
            color = colors[wire_idx]
            
            # 获取路径坐标
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            
            # 绘制路径线
            line = plt.plot(x_coords, y_coords, color=color, linewidth=2)[0]
            legend_elements.append((line, f'Path {wire_idx+1}'))
            
            # 绘制转折点（比线条小一些）
            plt.plot(x_coords[1:-1], y_coords[1:-1], 'o', 
                    color=color, markersize=2, alpha=0.6)
            
            # 绘制起点和终点（使用不同的标记，但大小适中）
            # 起点（圆形标记）
            start_point = plt.plot(path[0][0], path[0][1], 
                                's', color=color, markersize=10)[0]
            legend_elements.append((start_point, f'Start {wire_idx+1}'))
            
            # 终点（方形标记）
            end_point = plt.plot(path[-1][0], path[-1][1], 
                            's', color=color, markersize=10)[0]
            legend_elements.append((end_point, f'End {wire_idx+1}'))
        
        # 绘制交叉点（使用更小的标记）
        for point in self.cross_points:
            plt.plot(point["x"], point["y"], 'rx', markersize=2, alpha=0.7)
        
        # 绘制推荐转折点
        for point in self.recommended_turns:
            plt.plot(point["x"], point["y"], 'ro', markersize=2, alpha=0.7)
        
        # 设置图例，使用两列显示
        legend_handles, legend_labels = zip(*legend_elements)
        plt.legend(legend_handles, legend_labels, 
                bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
        
        plt.title(f'PCB Routing Visualization ({self.size}x{self.size} Grid)')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 设置坐标轴范围
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        
        # 保持横纵比例相等
        ax.set_aspect('equal')
        
        plt.show()

    def route_all_wires(self, wires: List[List[Tuple[int, int]]]) -> bool:
        self.colors = list(mcolors.TABLEAU_COLORS.values())[:len(wires)]
        
        # 收集所有端点
        for wire in wires:
            for point in wire:
                self.endpoints.add(tuple(point))  # 确保转换为tuple
        
        success = True
        total_path_length = 0.0

        for i, wire in enumerate(wires):
            print(f"\n开始布线第{i+1}条线路")
            
            # 直接获取起点和终点
            start = tuple(wire[0])  # 转换为tuple
            end = tuple(wire[1])    # 转换为tuple
            print(f"  处理线段: {start} -> {end}")
            
            max_attempts = 3
            path = None
            
            for attempt in range(max_attempts):
                path = self.a_star_with_llm(start, end)
                
                if path:
                    print(f"  线段布线成功，路径长度：{len(path)}")
                    total_path_length += len(path)
                    self.update_grid_and_weights(path)
                    self.paths.append(path)
                    break
                    
                if attempt < max_attempts - 1:
                    print(f"  线段第{attempt+1}次尝试失败，正在重试...")
                    # 在重试前临时降低权重
                    for pos in list(self.occupied_weights.keys()):
                        if pos not in self.endpoints:
                            self.occupied_weights[pos] *= 0.8
            
            if not path:
                print(f"  线段布线失败！")
                success = False
                break
                
            # API调用间隔
            if i < len(wires) - 1:
                sleep(1)
        
        # 可视化结果
        print(f"\n总布线长度: {total_path_length:.2f} ")
        self.visualize_routing(wires)
        return success
    
    def update_grid_and_weights(self, path: List[Tuple[int, int]]) -> None:
        """更新网格和权重信息"""
        for i, (x, y) in enumerate(path):
            self.grid[x][y] = 1
            pos = (x, y)
            
            if pos in self.endpoints:
                self.occupied_weights[pos] = 1.2  # 端点使用较低权重
            else:
                self.occupied_weights[pos] = 5.0  # 非端点使用较高权重
                
                # 在非端点周围添加渐变权重
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        npos = (nx, ny)
                        if (0 <= nx < self.size and 0 <= ny < self.size 
                            and npos not in self.endpoints):
                            distance = math.sqrt(dx*dx + dy*dy)
                            weight = 3.0 * math.exp(-distance/1.5)
                            self.occupied_weights[npos] = max(self.occupied_weights.get(npos, 0), weight)
                            