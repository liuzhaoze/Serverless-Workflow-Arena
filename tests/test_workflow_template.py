"""test_workflow_template.py

测试 src/serverless_workflow_arena/workflow_template.py 中的 WorkflowTemplate 类
"""

import json
import tempfile
from pathlib import Path

import pytest
from pytest import FixtureRequest

from serverless_workflow_arena.tools.dax_parser import EdgeInfo, NodeInfo
from serverless_workflow_arena.workflow_template import JsonContent, WorkflowTemplate


def _build_json_content(
    nodes: list[NodeInfo] | None = None,
    edges: list[EdgeInfo] | None = None,
    parallelisms: list[dict[str, int]] | None = None,
    memory_reqs: list[dict[str, int]] | None = None,
) -> JsonContent:
    """构建一个合法的 JsonContent 字典"""
    return {
        "nodes": nodes if nodes is not None else [],
        "edges": edges if edges is not None else [],
        "parallelisms": parallelisms if parallelisms is not None else [],
        "memory_reqs": memory_reqs if memory_reqs is not None else [],
    }


class TestWorkflowTemplate:
    """测试 WorkflowTemplate 类"""

    @pytest.fixture(params=["data/CYBERSHAKE.n.200.0.dax", "data/MONTAGE.n.200.0.dax"])
    def dax_file(self, request: FixtureRequest) -> Path:
        """用于测试的 DAX 文件"""
        return Path(__file__).parent / request.param

    def test_workflow_template_initialization_success(self, dax_file: Path):
        """测试 WorkflowTemplate 初始化成功"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        template = WorkflowTemplate(str(dax_file), 1000)

        assert len(template.computations) > 0
        assert len(template.memory_reqs) > 0
        assert len(template.parallelisms) > 0
        assert len(template.names) > 0
        assert (
            len(template.computations) == len(template.memory_reqs) == len(template.parallelisms) == len(template.names)
        )

        assert all(comp >= 0 for comp in template.computations)
        assert all(mem > 0 for mem in template.memory_reqs)
        assert all(par > 0 for par in template.parallelisms)

    def test_workflow_template_invalid_file_extension(self, dax_file: Path):
        """测试无效的文件扩展名"""
        non_dax_file = dax_file.with_suffix(".txt")
        non_dax_file.write_text("test content")

        with pytest.raises(ValueError, match="DAX file must have .dax suffix"):
            WorkflowTemplate(str(non_dax_file), 1000)

        non_dax_file.unlink()

    def test_workflow_template_negative_single_core_speed(self, dax_file: Path):
        """测试负的单核速度"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        with pytest.raises(ValueError, match="Single core speed must be positive"):
            WorkflowTemplate(str(dax_file), -1000)

    def test_workflow_template_single_core_speed_zero(self, dax_file: Path):
        """测试零单核速度"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        with pytest.raises(ValueError, match="Single core speed must be positive"):
            WorkflowTemplate(str(dax_file), 0)

    def test_workflow_template_memory_id_continuity(self):
        """测试内存需求 ID 不连续"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dax_file = Path(temp_dir) / "test.dax"
            dax_file.write_text("<adag/>", encoding="utf-8")

            json_file = Path(temp_dir) / "test.json"
            # memory_reqs 的 ID 从 1 开始，缺少 0
            data = _build_json_content(
                nodes=[{"id": 0, "runtime": 1.0, "name": "a"}, {"id": 1, "runtime": 1.0, "name": "b"}],
                memory_reqs=[{"id": 1, "value": 128}],
                parallelisms=[{"id": 0, "value": 1}, {"id": 1, "value": 1}],
            )
            json_file.write_text(json.dumps(data), encoding="utf-8")

            with pytest.raises(ValueError, match="Memory requirements ID not continuous: expected 0, got 1"):
                WorkflowTemplate(str(dax_file), 1000)

    def test_workflow_template_parallelism_id_continuity(self):
        """测试并行度 ID 不连续"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dax_file = Path(temp_dir) / "test.dax"
            dax_file.write_text("<adag/>", encoding="utf-8")

            json_file = Path(temp_dir) / "test.json"
            # parallelisms 缺少 ID=1
            data = _build_json_content(
                nodes=[{"id": 0, "runtime": 1.0, "name": "a"}, {"id": 1, "runtime": 1.0, "name": "b"}],
                memory_reqs=[{"id": 0, "value": 128}, {"id": 1, "value": 128}],
                parallelisms=[{"id": 0, "value": 1}, {"id": 2, "value": 1}],
            )
            json_file.write_text(json.dumps(data), encoding="utf-8")

            with pytest.raises(ValueError, match="Parallelisms ID not continuous: expected 1, got 2"):
                WorkflowTemplate(str(dax_file), 1000)

    def test_workflow_template_dag_node_id_continuity(self):
        """测试 DAG 节点 ID 不匹配"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dax_file = Path(temp_dir) / "test.dax"
            dax_file.write_text("<adag/>", encoding="utf-8")

            json_file = Path(temp_dir) / "test.json"
            # 节点 ID 跳过了 2
            data = _build_json_content(
                nodes=[
                    {"id": 0, "runtime": 1.0, "name": "a"},
                    {"id": 1, "runtime": 1.0, "name": "b"},
                    {"id": 3, "runtime": 1.0, "name": "c"},
                ],
                memory_reqs=[{"id": 0, "value": 128}, {"id": 1, "value": 128}, {"id": 2, "value": 128}],
                parallelisms=[{"id": 0, "value": 1}, {"id": 1, "value": 1}, {"id": 2, "value": 1}],
            )
            json_file.write_text(json.dumps(data), encoding="utf-8")

            with pytest.raises(ValueError, match="DAG node ID not continuous: expected 2, got 3"):
                WorkflowTemplate(str(dax_file), 1000)

    def test_workflow_template_length_mismatch(self):
        """测试长度不匹配"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dax_file = Path(temp_dir) / "test.dax"
            dax_file.write_text("<adag/>", encoding="utf-8")

            json_file = Path(temp_dir) / "test.json"
            # nodes 只有 1 个，但 memory_reqs 和 parallelisms 有 3 个
            data = _build_json_content(
                nodes=[{"id": 0, "runtime": 1.0, "name": "a"}],
                memory_reqs=[{"id": 0, "value": 128}, {"id": 1, "value": 128}, {"id": 2, "value": 128}],
                parallelisms=[{"id": 0, "value": 1}, {"id": 1, "value": 1}, {"id": 2, "value": 1}],
            )
            json_file.write_text(json.dumps(data), encoding="utf-8")

            with pytest.raises(
                ValueError, match="Number of nodes in DAG JSON does not match length of memory_reqs and parallelisms"
            ):
                WorkflowTemplate(str(dax_file), 1000)

    def test_workflow_template_computation_calculation(self, dax_file: Path):
        """测试计算需求计算"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        template = WorkflowTemplate(str(dax_file), 2000)
        baseline_template = WorkflowTemplate(str(dax_file), 1000)

        # 计算需求应该与单核速度成正比
        for i, comp in enumerate(template.computations):
            assert comp == baseline_template.computations[i] * 2

    def test_workflow_template_edges_structure(self, dax_file: Path):
        """测试边的结构"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        template = WorkflowTemplate(str(dax_file), 1000)

        for edge in template.edges:
            assert len(edge) == 3
            assert isinstance(edge[0], int)  # parent ID
            assert isinstance(edge[1], int)  # child ID
            assert isinstance(edge[2], int)  # data transfer size
            assert edge[2] >= 0

    def test_workflow_template_empty_workflow(self):
        """测试空工作流模板"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dax_file = Path(temp_dir) / "empty.dax"
            dax_file.write_text("<adag/>", encoding="utf-8")

            json_file = Path(temp_dir) / "empty.json"
            empty_data = _build_json_content()
            json_file.write_text(json.dumps(empty_data), encoding="utf-8")

            template = WorkflowTemplate(str(dax_file), 1000)

            assert len(template.computations) == 0
            assert len(template.memory_reqs) == 0
            assert len(template.parallelisms) == 0
            assert len(template.names) == 0
            assert len(template.edges) == 0

    def test_workflow_template_json_file_format_errors(self):
        """测试 JSON 文件格式错误"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dax_file = Path(temp_dir) / "test.dax"
            dax_file.write_text("<adag/>", encoding="utf-8")

            json_file = Path(temp_dir) / "test.json"
            json_file.write_text("invalid json content", encoding="utf-8")

            with pytest.raises(json.JSONDecodeError):
                WorkflowTemplate(str(dax_file), 1000)

    def test_workflow_template_missing_json_keys(self):
        """测试 JSON 文件缺少必需的键"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dax_file = Path(temp_dir) / "test.dax"
            dax_file.write_text("<adag/>", encoding="utf-8")

            json_file = Path(temp_dir) / "test.json"
            # 缺少 memory_reqs 键
            data = {"nodes": [], "edges": [], "parallelisms": []}  # type: ignore
            json_file.write_text(json.dumps(data), encoding="utf-8")

            with pytest.raises(KeyError):
                WorkflowTemplate(str(dax_file), 1000)

    def test_workflow_template_load_from_dax(self, dax_file: Path):
        """测试从 DAX 文件加载（无对应 .json 文件时）"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        json_file = dax_file.with_suffix(".json")
        if json_file.exists():
            pytest.skip(f"JSON file already exists: {json_file}")

        template = WorkflowTemplate(str(dax_file), 1000)

        # 从 DAX 加载时应使用默认值
        assert all(mem == 128 for mem in template.memory_reqs)
        assert all(par == 1 for par in template.parallelisms)

    def test_workflow_template_save_and_load_json(self, dax_file: Path):
        """测试 save_to_json 保存后再加载"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        # 先创建模板（从 DAX 加载）
        original = WorkflowTemplate(str(dax_file), 1000)

        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建一个假的 DAX 文件，WorkflowTemplate 会查找同名的 .json 文件
            fake_dax = Path(temp_dir) / "fake.dax"
            fake_dax.write_text("<adag/>", encoding="utf-8")

            # 保存 JSON 到与 DAX 同名的路径
            json_path = fake_dax.with_suffix(".json")
            original.save_to_json(str(json_path))

            # 验证 JSON 文件包含所有必需字段
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "nodes" in data
            assert "edges" in data
            assert "parallelisms" in data
            assert "memory_reqs" in data

            # 从 JSON 加载
            loaded = WorkflowTemplate(str(fake_dax), 1000)

            assert loaded.names == original.names
            assert len(loaded.computations) == len(original.computations)
            assert loaded.memory_reqs == original.memory_reqs
            assert loaded.parallelisms == original.parallelisms
