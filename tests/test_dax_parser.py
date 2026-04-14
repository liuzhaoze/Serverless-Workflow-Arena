"""test_dax_parser.py

测试 src/serverless_workflow_arena/tools/dax_parser.py 中的函数
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest import FixtureRequest

from serverless_workflow_arena.tools.dax_parser import JobInfo, calculate_data_transfer_size, parse_dax


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

    def test_parse_dax_success(self, dax_file: Path):
        """测试成功解析 DAX 文件"""
        parse_dax(str(dax_file))

        # 检查生成的 JSON 文件
        json_path = dax_file.with_suffix(".dag.json")
        assert json_path.exists()

        # 验证 JSON 内容
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 200

        # 验证节点数据结构
        for node in data["nodes"]:
            assert "id" in node
            assert "runtime" in node
            assert "name" in node
            assert isinstance(node["id"], int)
            assert isinstance(node["runtime"], (int, float))
            assert isinstance(node["name"], str)

        # 验证边数据结构
        for edge in data["edges"]:
            assert "parent" in edge
            assert "child" in edge
            assert "size_bytes" in edge
            assert isinstance(edge["parent"], int)
            assert isinstance(edge["child"], int)
            assert isinstance(edge["size_bytes"], int)

        # 清理文件
        json_path.unlink()

    def test_parse_dax_skip_existing_json(self, dax_file: Path):
        """测试跳过已存在的 JSON 文件"""
        # 首先解析 DAX 文件
        parse_dax(str(dax_file))

        json_path = dax_file.with_suffix(".dag.json")
        assert json_path.exists()

        # 记录原始修改时间和内容
        original_mtime = json_path.stat().st_mtime
        original_content = json_path.read_text(encoding="utf-8")

        with patch("builtins.print") as mock_print:
            parse_dax(str(dax_file))
            mock_print.assert_any_call(f"JSON file already exists: {json_path}, skipping parsing.")

        # 验证文件没有被修改
        assert json_path.stat().st_mtime == original_mtime
        assert json_path.read_text(encoding="utf-8") == original_content

        # 清理文件
        json_path.unlink()

    @patch("builtins.print")
    def test_parse_dax_prints_messages(self, mock_print: MagicMock, dax_file: Path):
        """测试 parse_dax 打印的消息"""
        parse_dax(str(dax_file))

        # 验证打印的消息
        expected_calls = [
            f"Parsing DAX file: {dax_file}",
            f"Found 200 jobs",
            f"Generated JSON file: {dax_file.with_suffix('.dag.json')}",
        ]

        call_args = [str(call[0][0]) for call in mock_print.call_args_list]

        assert any(expected_calls[0] in call for call in call_args)
        assert any(expected_calls[1] in call for call in call_args)
        assert any(expected_calls[2] in call for call in call_args)

        # 清理文件
        json_path = dax_file.with_suffix(".dag.json")
        if json_path.exists():
            json_path.unlink()

    def test_parse_dax_multiple_runs(self, dax_file: Path):
        """测试多次运行的行为"""
        # 第一次运行
        parse_dax(str(dax_file))

        json_path = dax_file.with_suffix(".dag.json")
        assert json_path.exists()

        # 记录第一次运行后的修改时间
        first_mtime = json_path.stat().st_mtime

        # 等待一小段时间以确保时间戳不同
        time.sleep(0.1)

        # 第二次运行应该跳过
        with patch("builtins.print") as mock_print:
            parse_dax(str(dax_file))
            mock_print.assert_any_call(f"JSON file already exists: {json_path}, skipping parsing.")

        # 验证文件没有被修改
        assert json_path.stat().st_mtime == first_mtime

        # 清理文件
        json_path.unlink()
