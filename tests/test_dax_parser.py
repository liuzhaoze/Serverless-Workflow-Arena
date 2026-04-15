"""test_dax_parser.py

测试 src/serverless_workflow_arena/tools/dax_parser.py 中的函数
"""

from pathlib import Path

import pytest
from pytest import FixtureRequest

from serverless_workflow_arena.tools.dax_parser import DagInfo, JobInfo, calculate_data_transfer_size, parse_dax


class TestCalculateDataTransferSize:
    """测试 calculate_data_transfer_size 函数"""

    def test_no_common_files(self):
        """测试没有共同文件的情况"""
        parent_job: JobInfo = {
            "id": 0,
            "runtime": 1.0,
            "name": "job_0",
            "files": {
                "file1.txt": {"size": 1024, "link": "output"},
                "file2.txt": {"size": 2048, "link": "output"},
            },
        }
        child_job: JobInfo = {
            "id": 1,
            "runtime": 1.0,
            "name": "job_1",
            "files": {
                "file3.txt": {"size": 512, "link": "input"},
                "file4.txt": {"size": 1024, "link": "input"},
            },
        }

        result = calculate_data_transfer_size(parent_job, child_job)
        assert result == 0

    def test_with_common_files(self):
        """测试有共同文件的情况"""
        parent_job: JobInfo = {
            "id": 0,
            "runtime": 1.0,
            "name": "job_0",
            "files": {
                "shared.txt": {"size": 1024, "link": "output"},
                "parent_only.txt": {"size": 2048, "link": "output"},
            },
        }
        child_job: JobInfo = {
            "id": 1,
            "runtime": 1.0,
            "name": "job_1",
            "files": {
                "shared.txt": {"size": 1024, "link": "input"},  # 大小相同
                "child_only.txt": {"size": 512, "link": "input"},
            },
        }

        result = calculate_data_transfer_size(parent_job, child_job)
        assert result == 1024  # 只有 shared.txt 是共同的

    def test_with_common_files_different_sizes(self):
        """测试共同文件大小不同的情况"""
        parent_job: JobInfo = {
            "id": 0,
            "runtime": 1.0,
            "name": "job_0",
            "files": {
                "shared.txt": {"size": 2048, "link": "output"},  # 父节点输出大小
                "parent_only.txt": {"size": 1024, "link": "output"},
            },
        }
        child_job: JobInfo = {
            "id": 1,
            "runtime": 1.0,
            "name": "job_1",
            "files": {
                "shared.txt": {"size": 1024, "link": "input"},  # 子节点输入大小不同
                "child_only.txt": {"size": 512, "link": "input"},
            },
        }

        result = calculate_data_transfer_size(parent_job, child_job)
        assert result == 2048  # 应该使用父节点的输出大小

    def test_multiple_common_files(self):
        """测试多个共同文件的情况"""
        parent_job: JobInfo = {
            "id": 0,
            "runtime": 1.0,
            "name": "job_0",
            "files": {
                "file1.txt": {"size": 1024, "link": "output"},
                "file2.txt": {"size": 2048, "link": "output"},
                "file3.txt": {"size": 512, "link": "output"},
            },
        }
        child_job: JobInfo = {
            "id": 1,
            "runtime": 1.0,
            "name": "job_1",
            "files": {
                "file1.txt": {"size": 1024, "link": "input"},
                "file2.txt": {"size": 2048, "link": "input"},
                "child_only.txt": {"size": 256, "link": "input"},
            },
        }

        result = calculate_data_transfer_size(parent_job, child_job)
        assert result == 1024 + 2048  # file1.txt + file2.txt

    def test_empty_files(self):
        """测试空文件列表的情况"""
        parent_job: JobInfo = {"id": 0, "runtime": 1.0, "name": "job_0", "files": {}}
        child_job: JobInfo = {"id": 1, "runtime": 1.0, "name": "job_1", "files": {}}

        result = calculate_data_transfer_size(parent_job, child_job)
        assert result == 0

    def test_parent_no_output_files(self):
        """测试父节点没有输出文件的情况"""
        parent_job: JobInfo = {
            "id": 0,
            "runtime": 1.0,
            "name": "job_0",
            "files": {
                "file1.txt": {"size": 1024, "link": "input"},  # 只有输入文件
            },
        }
        child_job: JobInfo = {
            "id": 1,
            "runtime": 1.0,
            "name": "job_1",
            "files": {
                "file1.txt": {"size": 1024, "link": "input"},
            },
        }

        result = calculate_data_transfer_size(parent_job, child_job)
        assert result == 0

    def test_child_no_input_files(self):
        """测试子节点没有输入文件的情况"""
        parent_job: JobInfo = {
            "id": 0,
            "runtime": 1.0,
            "name": "job_0",
            "files": {
                "file1.txt": {"size": 1024, "link": "output"},
            },
        }
        child_job: JobInfo = {
            "id": 1,
            "runtime": 1.0,
            "name": "job_1",
            "files": {
                "file1.txt": {"size": 1024, "link": "output"},  # 只有输出文件
            },
        }

        result = calculate_data_transfer_size(parent_job, child_job)
        assert result == 0


class TestParseDAX:
    """测试 parse_dax 函数"""

    # 测试数据目录的绝对路径
    DATA_DIR = Path(__file__).parent / "data"

    @pytest.fixture(params=["CYBERSHAKE.n.200.0.dax", "MONTAGE.n.200.0.dax"])
    def dax_file(self, request: FixtureRequest) -> Path:
        """用于测试的 DAX 文件"""
        return self.DATA_DIR / request.param

    def test_parse_dax_returns_dag_info(self, dax_file: Path):
        """测试 parse_dax 返回 DagInfo 字典"""
        result = parse_dax(str(dax_file))

        # 验证返回类型
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result

        # 验证节点数量
        assert len(result["nodes"]) == 200

        # 验证节点数据结构
        for node in result["nodes"]:
            assert "id" in node
            assert "runtime" in node
            assert "name" in node
            assert isinstance(node["id"], int)
            assert isinstance(node["runtime"], (int, float))
            assert isinstance(node["name"], str)

        # 验证边数据结构
        for edge in result["edges"]:
            assert "parent" in edge
            assert "child" in edge
            assert "size_bytes" in edge
            assert isinstance(edge["parent"], int)
            assert isinstance(edge["child"], int)
            assert isinstance(edge["size_bytes"], int)

    def test_parse_dax_nodes_sorted_by_id(self, dax_file: Path):
        """测试节点按 ID 排序"""
        result: DagInfo = parse_dax(str(dax_file))

        ids = [node["id"] for node in result["nodes"]]
        assert ids == sorted(ids)

    def test_parse_dax_edges_sorted(self, dax_file: Path):
        """测试边按 (parent, child) 排序"""
        result: DagInfo = parse_dax(str(dax_file))

        edge_keys = [(edge["parent"], edge["child"]) for edge in result["edges"]]
        assert edge_keys == sorted(edge_keys)

    def test_parse_dax_node_ids_continuous(self, dax_file: Path):
        """测试节点 ID 连续"""
        result: DagInfo = parse_dax(str(dax_file))

        ids = [node["id"] for node in result["nodes"]]
        assert ids == list(range(len(ids)))

    def test_parse_dax_edge_references_valid_nodes(self, dax_file: Path):
        """测试边引用的节点 ID 有效"""
        result: DagInfo = parse_dax(str(dax_file))

        node_ids = {node["id"] for node in result["nodes"]}
        for edge in result["edges"]:
            assert edge["parent"] in node_ids
            assert edge["child"] in node_ids
