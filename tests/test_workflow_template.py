"""test_workflow_template.py

测试 src/serverless_workflow_arena/workflow_template.py 中的 WorkflowTemplate 类
"""

import copy
import json
import tempfile
from pathlib import Path

import pytest
from pytest import FixtureRequest

from serverless_workflow_arena.tools.dax_parser import DagInfo
from serverless_workflow_arena.workflow_template import WorkflowTemplate


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

        single_core_speed = 1000

        try:
            template = WorkflowTemplate(str(dax_file), single_core_speed)

            # 基本验证 - 检查是否成功加载了数据
            assert len(template.computations) > 0
            assert len(template.memory_reqs) > 0
            assert len(template.parallelisms) > 0
            assert len(template.names) > 0
            assert len(template.computations) == len(template.memory_reqs) == len(template.parallelisms) == len(template.names)

            # 验证所有值都是非负数
            assert all(comp >= 0 for comp in template.computations)
            assert all(mem > 0 for mem in template.memory_reqs)
            assert all(par > 0 for par in template.parallelisms)

        except FileNotFoundError as e:
            pytest.skip(f"Required JSON files not found for {dax_file.name}: {e}")

    def test_workflow_template_invalid_file_extension(self, dax_file: Path):
        """测试无效的文件扩展名"""
        # 创建非 .dax 文件
        non_dax_file = dax_file.with_suffix(".txt")
        non_dax_file.write_text("test content")

        with pytest.raises(ValueError, match="DAX file must have .dax suffix"):
            WorkflowTemplate(str(non_dax_file), 1000)

        non_dax_file.unlink()  # 清理临时文件

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

    def test_workflow_template_memory_id_continuity(self, dax_file: Path):
        """测试内存需求 ID 不连续"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        memory_file = dax_file.with_suffix(".memory.json")
        if not memory_file.exists():
            pytest.skip(f"Memory JSON file not found: {memory_file}")

        # 读取原始数据
        with open(memory_file, "r", encoding="utf-8") as f:
            memory_data: dict[str, list[dict[str, int]]] = json.load(f)

        # 删除 0 号 ID，使 ID 与索引不匹配
        if len(memory_data["memory_reqs"]) <= 2:
            pytest.skip("Not enough memory requirement entries to test ID continuity")

        original_data = copy.deepcopy(memory_data)
        del memory_data["memory_reqs"][0]

        try:
            memory_file.write_text(json.dumps(memory_data), encoding="utf-8")
            with pytest.raises(ValueError, match="Memory requirements ID not continuous: expected 0, got 1"):
                WorkflowTemplate(str(dax_file), 1000)
        finally:
            # 恢复原始数据
            memory_file.write_text(json.dumps(original_data), encoding="utf-8")

    def test_workflow_template_parallelism_id_continuity(self, dax_file: Path):
        """测试并行度 ID 不连续"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        parallelism_file = dax_file.with_suffix(".parallelism.json")
        if not parallelism_file.exists():
            pytest.skip(f"Parallelism JSON file not found: {parallelism_file}")

        # 读取原始数据
        with open(parallelism_file, "r", encoding="utf-8") as f:
            parallelism_data: dict[str, list[dict[str, int]]] = json.load(f)

        # 删除 10 号 ID，使 ID 与索引不匹配
        if len(parallelism_data["parallelisms"]) <= 12:
            pytest.skip("Not enough parallelism entries to test ID continuity")

        original_data = copy.deepcopy(parallelism_data)
        del parallelism_data["parallelisms"][10]

        try:
            parallelism_file.write_text(json.dumps(parallelism_data), encoding="utf-8")
            with pytest.raises(ValueError, match="Parallelisms ID not continuous: expected 10, got 11"):
                WorkflowTemplate(str(dax_file), 1000)
        finally:
            # 恢复原始数据
            parallelism_file.write_text(json.dumps(original_data), encoding="utf-8")

    def test_workflow_template_dag_node_id_continuity(self, dax_file: Path):
        """测试 DAG 节点 ID 不匹配"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        dag_file = dax_file.with_suffix(".dag.json")
        if not dag_file.exists():
            pytest.skip(f"DAG JSON file not found: {dag_file}")

        # 读取原始数据
        with open(dag_file, "r", encoding="utf-8") as f:
            dag_data: DagInfo = json.load(f)

        # 修改 DAG 数据，使节点 ID 不连续
        if len(dag_data["nodes"]) <= 5:
            pytest.skip("Not enough nodes to test ID continuity")

        original_data = copy.deepcopy(dag_data)
        del dag_data["nodes"][2]
        del dag_data["nodes"][2]

        try:
            dag_file.write_text(json.dumps(dag_data), encoding="utf-8")
            with pytest.raises(ValueError, match="DAG node ID not continuous: expected 2, got 4"):
                WorkflowTemplate(str(dax_file), 1000)
        finally:
            # 恢复原始数据
            dag_file.write_text(json.dumps(original_data), encoding="utf-8")

    def test_workflow_template_length_mismatch(self, dax_file: Path):
        """测试长度不匹配"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        dag_file = dax_file.with_suffix(".dag.json")
        if not dag_file.exists():
            pytest.skip(f"DAG JSON file not found: {dag_file}")

        # 读取原始数据
        with open(dag_file, "r", encoding="utf-8") as f:
            dag_data: DagInfo = json.load(f)

        # 修改 DAG 数据，使节点数量与其他文件不匹配
        original_data = dag_data.copy()
        if len(dag_data["nodes"]) > 3:
            dag_data["nodes"] = dag_data["nodes"][:3]  # 减少节点数量

            try:
                dag_file.write_text(json.dumps(dag_data), encoding="utf-8")
                with pytest.raises(
                    ValueError,
                    match="Number of nodes in DAG JSON does not match length of memory_reqs and parallelisms",
                ):
                    WorkflowTemplate(str(dax_file), 1000)
            finally:
                # 恢复原始数据
                dag_file.write_text(json.dumps(original_data), encoding="utf-8")
        else:
            pytest.skip("Not enough nodes to test length mismatch")

    def test_workflow_template_computation_calculation(self, dax_file: Path):
        """测试计算需求计算"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        single_core_speed = 2000  # 不同的单核速度

        try:
            template = WorkflowTemplate(str(dax_file), single_core_speed)

            # 验证计算需求随着单核速度变化而正确变化
            # 首先用基准速度创建一个模板
            baseline_template = WorkflowTemplate(str(dax_file), 1000)

            # 计算需求应该与单核速度成正比
            for i, comp in enumerate(template.computations):
                expected_comp = baseline_template.computations[i] * 2  # 2000 vs 1000，所以是2倍
                assert comp == expected_comp

        except FileNotFoundError as e:
            pytest.skip(f"Required JSON files not found for {dax_file.name}: {e}")

    def test_workflow_template_edges_structure(self, dax_file: Path):
        """测试边的结构"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        try:
            template = WorkflowTemplate(str(dax_file), 1000)

            # 验证边的格式
            for edge in template.edges:
                assert len(edge) == 3
                assert isinstance(edge[0], int)  # parent ID
                assert isinstance(edge[1], int)  # child ID
                assert isinstance(edge[2], int)  # data transfer size
                assert edge[2] >= 0  # 传输大小应该非负

        except FileNotFoundError as e:
            pytest.skip(f"Required JSON files not found for {dax_file.name}: {e}")

    def test_workflow_template_empty_workflow(self):
        """测试空工作流模板"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 创建空的 DAG JSON 文件
            empty_dag_data: DagInfo = {"nodes": [], "edges": []}

            # 创建空的内存需求 JSON 文件
            empty_memory_data: dict[str, list[dict[str, int]]] = {"memory_reqs": []}

            # 创建空的并行度 JSON 文件
            empty_parallelism_data: dict[str, list[dict[str, int]]] = {"parallelisms": []}

            # 写入文件
            dax_file = temp_path / "empty.dax"
            dag_file = temp_path / "empty.dag.json"
            memory_file = temp_path / "empty.memory.json"
            parallelism_file = temp_path / "empty.parallelism.json"

            dag_file.write_text(json.dumps(empty_dag_data), encoding="utf-8")
            memory_file.write_text(json.dumps(empty_memory_data), encoding="utf-8")
            parallelism_file.write_text(json.dumps(empty_parallelism_data), encoding="utf-8")

            template = WorkflowTemplate(str(dax_file), 1000)

            assert len(template.computations) == 0
            assert len(template.memory_reqs) == 0
            assert len(template.parallelisms) == 0
            assert len(template.names) == 0
            assert len(template.edges) == 0

    def test_workflow_template_json_file_format_errors(self, dax_file: Path):
        """测试 JSON 文件格式错误"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        memory_file = dax_file.with_suffix(".memory.json")
        if not memory_file.exists():
            pytest.skip(f"Memory JSON file not found: {memory_file}")

        # 读取原始数据
        with open(memory_file, "r", encoding="utf-8") as f:
            original_data = f.read()

        try:
            # 测试损坏的 JSON 文件
            memory_file.write_text("invalid json content", encoding="utf-8")

            with pytest.raises(json.JSONDecodeError):
                WorkflowTemplate(str(dax_file), 1000)
        finally:
            # 恢复原始数据
            memory_file.write_text(original_data, encoding="utf-8")

    def test_workflow_template_missing_json_keys(self, dax_file: Path):
        """测试 JSON 文件缺少必需的键"""
        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        memory_file = dax_file.with_suffix(".memory.json")
        if not memory_file.exists():
            pytest.skip(f"Memory JSON file not found: {memory_file}")

        # 读取原始数据
        with open(memory_file, "r", encoding="utf-8") as f:
            original_data = f.read()

        try:
            # 测试缺少 memory_reqs 键
            memory_data: dict[str, list[dict[str, int]]] = {"wrong_key": []}
            memory_file.write_text(json.dumps(memory_data), encoding="utf-8")

            with pytest.raises(KeyError):
                WorkflowTemplate(str(dax_file), 1000)
        finally:
            # 恢复原始数据
            memory_file.write_text(original_data, encoding="utf-8")
