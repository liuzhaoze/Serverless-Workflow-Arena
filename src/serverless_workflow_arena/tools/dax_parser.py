"""dax_parser.py

解析 Pegasus DAX 文件，将其导出为 JSON 文件：

{
    "nodes": [
        {"id": 0, "runtime": 0.31, "name": "ExtractSGT"},
        {"id": 1, "runtime": 0.45, "name": "SeismogramSynthesis"},
        ...
    ],
    "edges": [
        {"parent": 0, "child": 1, "size_bytes": 2048},
        {"parent": 1, "child": 2, "size_bytes": 39277786880},
        ...
    ]
}

说明：
- runtime 来自 <job runtime="..."> (单位：秒)
- name 来自 <job name="...">
- size_bytes = 父节点输出 & 子节点输入的共同文件 size 之和 (单位：字节)
"""

from typing import Literal, TypedDict, cast

LinkType = Literal["input", "output"]


class FileInfo(TypedDict):
    link: LinkType
    size: int


class JobInfo(TypedDict):
    id: int
    runtime: float
    name: str
    files: dict[str, FileInfo]


class NodeInfo(TypedDict):
    id: int
    runtime: float
    name: str


class EdgeInfo(TypedDict):
    parent: int
    child: int
    size_bytes: int


class DagInfo(TypedDict):
    nodes: list[NodeInfo]
    edges: list[EdgeInfo]


import xml.etree.ElementTree as ET
from pathlib import Path

from .pretty_json import pretty_json_dump


def parse_dax(path: str) -> str:
    """解析 DAX 文件并在同目录下对应的 JSON 文件

    如果 JSON 文件已存在，则跳过解析。

    Args:
        path (str): DAX 文件路径

    Returns:
        str: 生成的 DAG JSON 文件路径
    """

    dax_path: Path = Path(path)
    print(f"Parsing DAX file: {dax_path}")

    json_path = dax_path.with_suffix(".dag.json")
    if json_path.exists():
        print(f"JSON file already exists: {json_path}, skipping parsing.")
        return str(json_path)

    tree = ET.parse(dax_path)
    root = tree.getroot()

    # 从 root tag 中获取 namespace (此处为 http://pegasus.isi.edu/schema/DAX)
    ns = {"dax": root.tag.split("}")[0][1:]} if "}" in root.tag else {}

    # 解析所有 job 的 runtime 和文件IO信息
    jobs: dict[str, JobInfo] = {}
    job_xpath = ".//dax:job" if ns else ".//job"
    uses_xpath = ".//dax:uses" if ns else ".//uses"

    for job in root.findall(job_xpath, ns):
        if (job_id := job.get("id")) is None:
            continue
        if (job_runtime := job.get("runtime")) is None:
            continue
        if (job_name := job.get("name")) is None:
            continue

        jobs[job_id] = {
            "id": int(job_id.replace("ID", "")),
            "runtime": float(job_runtime),
            "name": job_name,
            "files": {},
        }

        # 解析文件IO信息
        for uses in job.findall(uses_xpath, ns):
            if (file_name := uses.get("file")) is None:
                continue
            if (link_type := uses.get("link")) not in {"input", "output"}:
                continue
            if (file_size := uses.get("size")) is None:
                continue

            jobs[job_id]["files"][file_name] = {"link": cast(LinkType, link_type), "size": int(file_size)}

    # 解析 job 的依赖关系
    edges: list[EdgeInfo] = []
    child_xpath = ".//dax:child" if ns else ".//child"
    parent_xpath = ".//dax:parent" if ns else ".//parent"

    for child in root.findall(child_xpath, ns):
        if (child_id := child.get("ref")) is None:
            continue

        for parent in child.findall(parent_xpath, ns):
            if (parent_id := parent.get("ref")) is None:
                continue

            size_bytes = calculate_data_transfer_size(jobs[parent_id], jobs[child_id])

            edges.append(
                {
                    "parent": int(parent_id.replace("ID", "")),
                    "child": int(child_id.replace("ID", "")),
                    "size_bytes": size_bytes,
                }
            )

    # 生成结果
    result: DagInfo = {
        "nodes": [
            {"id": job_data["id"], "runtime": job_data["runtime"], "name": job_data["name"]}
            for job_data in jobs.values()
        ],
        "edges": edges,
    }

    # 按 ID 排序
    result["nodes"].sort(key=lambda x: x["id"])
    result["edges"].sort(key=lambda x: (x["parent"], x["child"]))

    # 写入 JSON 文件
    pretty_json_dump(str(json_path), nodes=result["nodes"], edges=result["edges"])

    print(f"Found {len(result['nodes'])} jobs and {len(result['edges'])} dependencies")
    print(f"Generated JSON file: {json_path}")
    return str(json_path)


def calculate_data_transfer_size(parent_job: JobInfo, child_job: JobInfo) -> int:
    """计算父节点到子节点的数据传输量

    父节点的输出文件与子节点的输入文件的交集中的文件的大小之和

    Args:
        parent_job (JobInfo): 父节点的 job 信息
        child_job (JobInfo): 子节点的 job 信息

    Returns:
        int: 传输数据量 (单位：字节)
    """
    parent_outputs: dict[str, int] = {}
    child_inputs: dict[str, int] = {}

    # 收集父节点的输出文件
    for file_name, file_info in parent_job["files"].items():
        if file_info["link"] == "output":
            parent_outputs[file_name] = file_info["size"]

    # 收集子节点的输入文件
    for file_name, file_info in child_job["files"].items():
        if file_info["link"] == "input":
            child_inputs[file_name] = file_info["size"]

    # 计算共同文件的大小之和
    common_files = set(parent_outputs.keys()) & set(child_inputs.keys())

    # 对于同一个文件，父节点的输出大小和子节点的输入大小可能不一致
    # 这里以父节点的输出大小为准
    total_size = sum(parent_outputs[file] for file in common_files)

    assert total_size >= 0
    return total_size
