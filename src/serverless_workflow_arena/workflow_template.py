"""workflow_template.py

serverless工作流模板
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from .tools.dax_parser import EdgeInfo, NodeInfo, parse_dax
from .tools.pretty_json import pretty_json_dump

DEFAULT_PARALLELISM = 1
DEFAULT_MEMORY_REQ = 128


class JsonContent(TypedDict):
    nodes: list[NodeInfo]
    edges: list[EdgeInfo]
    parallelisms: list[dict[str, int]]
    memory_reqs: list[dict[str, int]]


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
    names: tuple[str, ...]
    runtimes: tuple[float, ...]
    computations: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]

    def __init__(self, dax_path: str, single_core_speed: int):
        if Path(dax_path).suffix != ".dax":
            raise ValueError(f"DAX file must have .dax suffix: {dax_path}")
        if single_core_speed <= 0:
            raise ValueError(f"Single core speed must be positive: {single_core_speed}")
        if single_core_speed < 100:
            print("Warning: single core speed is recommended to be not less than 100")

        json_path = Path(dax_path).with_suffix(".json")

        if json_path.exists():
            print(f"检测到已解析的 JSON 文件：{json_path}，直接从 JSON 文件加载工作流模板")
            self._load_from_json(str(json_path), single_core_speed)
        else:
            print(f"从 DAX 文件解析工作流模板：{dax_path}，并使用默认并行度和内存需求")
            self._load_from_dax(dax_path, single_core_speed)

    def _load_from_dax(self, dax_path: str, single_core_speed: int):
        dag_info = parse_dax(dax_path)
        nodes_sorted = sorted(dag_info["nodes"], key=lambda x: x["id"])
        for index, item in enumerate(nodes_sorted):
            if item["id"] != index:
                raise ValueError(f"DAG node ID not continuous: expected {index}, got {item['id']}")

        self.memory_reqs = tuple(DEFAULT_MEMORY_REQ for _ in nodes_sorted)
        self.parallelisms = tuple(DEFAULT_PARALLELISM for _ in nodes_sorted)
        self.names = tuple(node["name"] for node in nodes_sorted)
        self.runtimes = tuple(node["runtime"] for node in nodes_sorted)
        self.computations = tuple(
            int(round(r * single_core_speed * p)) for r, p in zip(self.runtimes, self.parallelisms)
        )
        self.edges = tuple((edge["parent"], edge["child"], edge["size_bytes"]) for edge in dag_info["edges"])

    def _load_from_json(self, json_path: str, single_core_speed: int):
        with open(json_path, "r", encoding="utf-8") as f:
            data: JsonContent = json.load(f)

        # 验证并读取内存需求
        memory_reqs_sorted = sorted(data["memory_reqs"], key=lambda x: x["id"])
        for index, item in enumerate(memory_reqs_sorted):
            if item["id"] != index:
                raise ValueError(f"Memory requirements ID not continuous: expected {index}, got {item['id']}")
        self.memory_reqs = tuple(item["value"] for item in memory_reqs_sorted)

        # 验证并读取并行度
        parallelisms_sorted = sorted(data["parallelisms"], key=lambda x: x["id"])
        for index, item in enumerate(parallelisms_sorted):
            if item["id"] != index:
                raise ValueError(f"Parallelisms ID not continuous: expected {index}, got {item['id']}")
        self.parallelisms = tuple(item["value"] for item in parallelisms_sorted)

        # 验证并读取DAG信息
        nodes_sorted = sorted(data["nodes"], key=lambda x: x["id"])
        for index, item in enumerate(nodes_sorted):
            if item["id"] != index:
                raise ValueError(f"DAG node ID not continuous: expected {index}, got {item['id']}")

        if not (len(nodes_sorted) == len(self.memory_reqs) == len(self.parallelisms)):
            raise ValueError("Number of nodes in DAG JSON does not match length of memory_reqs and parallelisms")

        self.names = tuple(node["name"] for node in nodes_sorted)
        self.runtimes = tuple(node["runtime"] for node in nodes_sorted)
        self.computations = tuple(
            int(round(r * single_core_speed * p)) for r, p in zip(self.runtimes, self.parallelisms)
        )
        self.edges = tuple((edge["parent"], edge["child"], edge["size_bytes"]) for edge in data["edges"])

    def save_to_json(self, json_path: str):
        pretty_json_dump(
            path=json_path,
            nodes=[{"id": i, "runtime": r, "name": n} for i, (r, n) in enumerate(zip(self.runtimes, self.names))],
            edges=[{"parent": p, "child": c, "size_bytes": s} for p, c, s in self.edges],
            parallelisms=[{"id": i, "value": p} for i, p in enumerate(self.parallelisms)],
            memory_reqs=[{"id": i, "value": m} for i, m in enumerate(self.memory_reqs)],
        )
