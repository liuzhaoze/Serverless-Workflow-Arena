"""workflow_template.py

serverless工作流模板
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .tools import generate_memory_requirements, generate_parallelisms, parse_dax


@dataclass(slots=True)
class WorkflowTemplate:
    """serverless工作流模板

    serverless工作流模板保存了工作流的静态结构信息，可用于生成具体的工作流实例

    Args:
        dax_path (str): DAX 文件路径
        single_core_speed (int): 单核计算速度 (单个 CPU 核心每秒可执行的计算操作数量)，用于计算函数的 computation
    """

    memory_reqs: tuple[int, ...]
    parallelisms: tuple[int, ...]
    computations: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]

    def __init__(self, dax_path: str, single_core_speed: int):
        if Path(dax_path).suffix != ".dax":
            raise ValueError(f"DAX file must have .dax suffix: {dax_path}")
        if single_core_speed <= 0:
            raise ValueError(f"Single core speed must be positive: {single_core_speed}")
        if single_core_speed < 100:
            print("Warning: single core speed is recommended to be not less than 100")

        dag_path = parse_dax(dax_path)
        memory_path = generate_memory_requirements(dax_path)
        parallelism_path = generate_parallelisms(dax_path)

        # 读取内存需求
        with open(memory_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        memory_reqs = data["memory_reqs"]
        memory_reqs_sorted = sorted(memory_reqs, key=lambda x: x["id"])
        for index, item in enumerate(memory_reqs_sorted):
            if item["id"] != index:
                raise ValueError(f"Memory requirements ID not continuous: expected {index}, got {item['id']}")
        self.memory_reqs = tuple(item["value"] for item in memory_reqs_sorted)

        # 读取并行度
        with open(parallelism_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        parallelisms = data["parallelisms"]
        parallelisms_sorted = sorted(parallelisms, key=lambda x: x["id"])
        for index, item in enumerate(parallelisms_sorted):
            if item["id"] != index:
                raise ValueError(f"Parallelisms ID not continuous: expected {index}, got {item['id']}")
        self.parallelisms = tuple(item["value"] for item in parallelisms_sorted)

        # 读取DAG信息
        with open(dag_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data["nodes"]
        nodes_sorted = sorted(nodes, key=lambda x: x["id"])
        for index, item in enumerate(nodes_sorted):
            if item["id"] != index:
                raise ValueError(f"DAG node ID not continuous: expected {index}, got {item['id']}")

        if not (len(nodes) == len(self.memory_reqs) == len(self.parallelisms)):
            raise ValueError("Number of nodes in DAG JSON does not match length of memory_reqs and parallelisms")

        self.computations = tuple(
            int(round(n["runtime"] * single_core_speed * p)) for n, p in zip(nodes_sorted, self.parallelisms)
        )
        self.edges = tuple((edge["parent"], edge["child"], edge["size_bytes"]) for edge in data["edges"])
