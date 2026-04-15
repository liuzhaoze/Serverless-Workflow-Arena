"""test_workflow.py

测试 src/serverless_workflow_arena/workflow.py 中的 Workflow 类和 SubmittableFunc 类
"""

from pathlib import Path

import pytest
import rustworkx as rx
from pytest import FixtureRequest

from serverless_workflow_arena.function import FuncStat
from serverless_workflow_arena.workflow import SubmittableFunc, Workflow
from serverless_workflow_arena.workflow_template import WorkflowTemplate


class TestSubmittableFunc:
    """测试 SubmittableFunc NamedTuple"""

    def test_submittable_func_creation(self):
        """测试 SubmittableFunc 创建"""
        func = SubmittableFunc(fn_id=1, submission_time=5.0)
        assert func.fn_id == 1
        assert func.submission_time == 5.0

    def test_submittable_func_immutability(self):
        """测试 SubmittableFunc 不可变性"""
        func = SubmittableFunc(fn_id=1, submission_time=5.0)

        # 应该不能修改属性
        with pytest.raises(AttributeError):
            func.fn_id = 2  # type: ignore


class TestWorkflow:
    """测试 Workflow 类"""

    @pytest.fixture(params=["data/CYBERSHAKE.n.200.0.dax", "data/MONTAGE.n.200.0.dax"])
    def workflow_template(self, request: FixtureRequest):
        """使用 WorkflowTemplate 加载真实的工作流数据"""
        dax_file: Path = Path(__file__).parent / request.param

        if not dax_file.exists():
            pytest.skip(f"DAX file not found: {dax_file}")

        try:
            return WorkflowTemplate(str(dax_file), single_core_speed=1000)
        except Exception as e:
            pytest.skip(f"Failed to load workflow template from {dax_file.name}: {e}")

    @pytest.fixture
    def workflow_from_template(self, workflow_template: WorkflowTemplate):
        """从 WorkflowTemplate 创建 Workflow 实例"""
        return Workflow(
            wf_id=0,
            arrival_time=0.0,
            computations=workflow_template.computations,
            memory_reqs=workflow_template.memory_reqs,
            parallelisms=workflow_template.parallelisms,
            edges=workflow_template.edges,
        )

    def test_workflow_initialization_success(self, workflow_from_template: Workflow):
        """测试工作流初始化成功"""
        workflow = workflow_from_template

        assert workflow.wf_id == 0
        assert workflow.arrival_time == 0.0
        assert len(workflow) > 0

        # 验证函数
        for i, fn in enumerate(workflow):
            assert fn.wf_id == 0
            assert fn.fn_id == i
            assert fn.state == FuncStat.PENDING

        # 验证源点函数存在（至少有一个源点）
        assert len(workflow.source_functions) >= 1

    def test_workflow_initialization_negative_arrival_time(self, workflow_template: WorkflowTemplate):
        """测试负的到达时间"""
        template = workflow_template
        with pytest.raises(ValueError, match="Arrival time must be non-negative"):
            Workflow(
                wf_id=0,
                arrival_time=-1.0,
                computations=template.computations,
                memory_reqs=template.memory_reqs,
                parallelisms=template.parallelisms,
                edges=template.edges,
            )

    def test_workflow_initialization_mismatched_lengths(self, workflow_template: WorkflowTemplate):
        """测试不匹配的数组长度"""
        template = workflow_template

        # 创建不匹配的内存需求长度
        mismatched_memory_reqs = template.memory_reqs[: len(template.memory_reqs) // 2]  # 取一半长度

        with pytest.raises(ValueError, match="Length of computations, memory_reqs, and parallelisms must be equal"):
            Workflow(
                wf_id=1,
                arrival_time=0.0,
                computations=template.computations,
                memory_reqs=mismatched_memory_reqs,
                parallelisms=template.parallelisms,
                edges=template.edges,
            )

    def test_workflow_initialization_invalid_edges(self, workflow_template: WorkflowTemplate):
        """测试无效的边格式"""
        template = workflow_template

        # 创建无效的边格式（缺少数据大小）
        invalid_edges = ((0, 1), (2, 3, 1024))  # 第一个边缺少数据大小

        with pytest.raises(ValueError, match="Each edge must be a tuple of"):
            Workflow(
                wf_id=0,
                arrival_time=0.0,
                computations=template.computations,
                memory_reqs=template.memory_reqs,
                parallelisms=template.parallelisms,
                edges=invalid_edges,  # type: ignore
            )

    def test_workflow_slots(self, workflow_from_template: Workflow):
        """测试 __slots__ 属性"""
        expected_slots = {
            "wf_id",
            "arrival_time",
            "_functions",
            "dag",
            "source_functions",
            "pending_functions",
            "submitted_functions",
            "running_functions",
            "finished_functions",
        }
        actual_slots = set(Workflow.__slots__)
        assert actual_slots == expected_slots

        # 测试不能添加新属性
        with pytest.raises(AttributeError):
            workflow_from_template.new_attribute = "test"  # type: ignore

    def test_reset(self, workflow_from_template: Workflow):
        """测试重置功能"""
        workflow = workflow_from_template

        # 获取一个源点函数进行测试
        source_fn_id = next(iter(workflow.source_functions))

        # 修改一些函数状态
        workflow[source_fn_id].submit(1.0)
        workflow[source_fn_id].run(2.0)
        workflow[source_fn_id].finish(5.0)

        # 修改状态集合
        original_pending = workflow.pending_functions.copy()
        workflow.pending_functions = original_pending - {source_fn_id}
        workflow.submitted_functions = {source_fn_id}
        workflow.running_functions = set()
        workflow.finished_functions = set()

        # 重置
        workflow.reset()

        # 验证所有函数都被重置
        for fn in workflow:
            assert fn.state == FuncStat.PENDING
            assert fn.submission_time is None
            assert fn.start_time is None
            assert fn.completion_time is None

        # 验证状态集合恢复到初始状态
        assert workflow.pending_functions == set(range(len(workflow)))
        assert workflow.submitted_functions == set()
        assert workflow.running_functions == set()
        assert workflow.finished_functions == set()

    def test_submit_function(self, workflow_from_template: Workflow):
        """测试提交函数"""
        workflow = workflow_from_template

        # 首先重置以确保状态集合正确初始化
        workflow.reset()

        # 提交一个源点函数
        source_fn_id = next(iter(workflow.source_functions))
        workflow.submit_function(source_fn_id, 1.0)

        assert workflow[source_fn_id].state == FuncStat.SUBMITTED
        assert workflow[source_fn_id].submission_time == 1.0
        assert source_fn_id not in workflow.pending_functions
        assert source_fn_id in workflow.submitted_functions

    def test_run_function(self, workflow_from_template: Workflow):
        """测试运行函数"""
        workflow = workflow_from_template

        # 首先重置
        workflow.reset()

        # 先提交一个源点函数
        source_fn_id = next(iter(workflow.source_functions))
        workflow.submit_function(source_fn_id, 1.0)

        # 然后运行
        workflow.run_function(source_fn_id, 2.0)

        assert workflow[source_fn_id].state == FuncStat.RUNNING
        assert workflow[source_fn_id].start_time == 2.0
        assert source_fn_id not in workflow.submitted_functions
        assert source_fn_id in workflow.running_functions

    def test_finish_function(self, workflow_from_template: Workflow):
        """测试完成函数"""
        workflow = workflow_from_template

        # 首先重置
        workflow.reset()

        # 先提交和运行一个源点函数
        source_fn_id = next(iter(workflow.source_functions))
        workflow.submit_function(source_fn_id, 1.0)
        workflow.run_function(source_fn_id, 2.0)

        # 然后完成
        workflow.finish_function(source_fn_id, 5.0)

        assert workflow[source_fn_id].state == FuncStat.FINISHED
        assert workflow[source_fn_id].completion_time == 5.0
        assert source_fn_id not in workflow.running_functions
        assert source_fn_id in workflow.finished_functions

    def test_get_submittable_functions_basic(self, workflow_from_template: Workflow):
        """测试基本可提交函数功能"""
        workflow = workflow_from_template
        workflow.reset()

        # 初始状态下无可提交函数（工作流需要显式提交源点）
        submittable = workflow.get_submittable_functions(0)
        assert len(submittable) == 0

        # 手动提交一个源点函数
        source_fn_id = next(iter(workflow.source_functions))
        workflow.submit_function(source_fn_id, workflow.arrival_time)
        workflow.run_function(source_fn_id, 1.0)
        workflow.finish_function(source_fn_id, 2.0)

        # 现在应该有可提交的函数（依赖于已完成的函数的函数）
        submittable = workflow.get_submittable_functions(source_fn_id)
        # 子函数的数量可能为0（如果源点函数没有后继），也可能大于0
        assert isinstance(submittable, list)

    def test_get_submittable_workflow_dependencies(self, workflow_from_template: Workflow):
        """测试工作流依赖关系"""
        workflow = workflow_from_template
        workflow.reset()

        # 提交并完成一个源点函数
        source_fn_id = next(iter(workflow.source_functions))
        workflow.submit_function(source_fn_id, workflow.arrival_time)
        workflow.run_function(source_fn_id, 1.0)
        workflow.finish_function(source_fn_id, 2.0)

        # 获取可提交函数
        submittable = workflow.get_submittable_functions(source_fn_id)

        # 验证所有返回的函数都是有效的（依赖于已完成的函数）
        for submittable_func in submittable:
            assert submittable_func.fn_id < len(workflow)
            assert submittable_func.submission_time == workflow[source_fn_id].completion_time

    def test_completed_property(self, workflow_from_template: Workflow):
        """测试工作流完成状态"""
        workflow = workflow_from_template
        workflow.reset()

        # 初始状态未完成
        assert not workflow.completed

        # 完成一个源点函数
        source_fn_id = next(iter(workflow.source_functions))
        workflow.submit_function(source_fn_id, 0.0)
        workflow.run_function(source_fn_id, 1.0)
        workflow.finish_function(source_fn_id, 3.0)

        # 工作流应该仍未完成（除非只有一个函数）
        if len(workflow) > 1:
            assert not workflow.completed

        # 完成所有函数（对于大型工作流，我们只测试前几个函数）
        for i in range(min(3, len(workflow))):
            if i != source_fn_id:  # 跳过已完成的源点函数
                workflow.submit_function(i, float(i))
                workflow.run_function(i, float(i + 1))
                workflow.finish_function(i, float(i + 2))

        # 验证完成状态（对于大型工作流，可能仍然未完成）
        # 这里我们只检查已完成属性能正确工作
        completed_count = sum(1 for fn in workflow if fn.state == FuncStat.FINISHED)
        if completed_count == len(workflow):
            assert workflow.completed
        else:
            assert not workflow.completed

    def test_source_functions_detection(self, workflow_from_template: Workflow):
        """测试源点函数检测"""
        workflow = workflow_from_template

        # 验证源点函数集合不为空
        assert len(workflow.source_functions) >= 1

        # 验证所有源点函数ID都是有效的
        for source_fn_id in workflow.source_functions:
            assert 0 <= source_fn_id < len(workflow)

        # 验证源点函数确实是工作流中的函数
        assert workflow.source_functions.issubset(set(range(len(workflow))))

    def test_single_function_workflow(self):
        """测试只有一个函数的工作流"""
        workflow = Workflow(
            wf_id=0, arrival_time=0.0, computations=(1000,), memory_reqs=(512,), parallelisms=(2,), edges=()  # 没有边
        )

        assert len(workflow) == 1
        assert workflow.source_functions == {0}

        # 这个函数应该可以直接提交（作为源点）
        workflow.reset()
        assert 0 in workflow.pending_functions

    def test_complex_workflow_with_multiple_sources(self):
        """测试有多个源点的工作流"""
        workflow = Workflow(
            wf_id=0,
            arrival_time=0.0,
            computations=(1000, 1500, 2000, 1200),
            memory_reqs=(512, 768, 1024, 640),
            parallelisms=(2, 3, 4, 2),
            edges=((2, 3, 1024),),  # 只有 2->3 的边，0、1、2都是源点
        )

        # 0、1、2都应该是源点（都没有前驱）
        assert workflow.source_functions == {0, 1, 2}

    def test_workflow_with_cycle_detection(self):
        """测试有循环的工作流应该失败"""
        # rustworkx 在创建工作流时会检查循环，有循环的工作流会抛出 DAGHasCycle 异常
        with pytest.raises(rx.DAGHasCycle):  # rustworkx.DAGHasCycle 是具体异常类型
            Workflow(
                wf_id=0,
                arrival_time=0.0,
                computations=(1000, 1500),
                memory_reqs=(512, 768),
                parallelisms=(2, 3),
                edges=((0, 1, 512), (1, 0, 256)),  # 0->1, 1->0 形成循环
            )

    def test_complete_workflow_execution(self, workflow_from_template: Workflow):
        """测试完整的工作流执行流程"""
        workflow = workflow_from_template
        workflow.reset()

        # 验证初始状态
        total_functions = len(workflow)
        assert workflow.pending_functions == set(range(total_functions))
        assert workflow.submitted_functions == set()
        assert workflow.running_functions == set()
        assert workflow.finished_functions == set()

        # 提交源点函数
        source_fns = list(workflow.source_functions)
        for fn_id in source_fns:
            workflow.submit_function(fn_id, workflow.arrival_time)

        # 验证提交后状态
        for fn_id in source_fns:
            assert fn_id in workflow.submitted_functions

        # 运行并完成一个源点函数
        test_fn_id = source_fns[0]
        workflow.run_function(test_fn_id, 1.0)
        assert test_fn_id in workflow.running_functions
        assert test_fn_id not in workflow.submitted_functions

        workflow.finish_function(test_fn_id, 3.0)
        assert test_fn_id in workflow.finished_functions
        assert test_fn_id not in workflow.running_functions

        # 获取可提交的函数
        submittable = workflow.get_submittable_functions(test_fn_id)
        # 验证返回的是有效的函数列表
        assert isinstance(submittable, list)
        for sub_fn in submittable:
            assert 0 <= sub_fn.fn_id < total_functions
            assert sub_fn.submission_time == workflow[test_fn_id].completion_time

    def test_workflow_template_data_validation(self, workflow_from_template: Workflow):
        """测试从 WorkflowTemplate 创建的工作流数据有效性"""
        workflow = workflow_from_template

        # 验证图结构基本有效性
        assert hasattr(workflow, "dag")
        assert len(workflow) > 0

        # 验证函数数据有效性
        for i, fn in enumerate(workflow):
            assert fn.fn_id == i
            assert fn.memory_req > 0
            assert fn.parallelism > 0
            assert fn.computation >= 0

        # 验证源点函数有效性
        for source_fn_id in workflow.source_functions:
            assert 0 <= source_fn_id < len(workflow)

        # 验证状态集合一致性
        workflow.reset()  # 重置以初始化状态集合
        total_functions = set(range(len(workflow)))
        assert workflow.pending_functions == total_functions
        assert workflow.submitted_functions == set()
        assert workflow.running_functions == set()
        assert workflow.finished_functions == set()

    def test_getitem(self, workflow_from_template: Workflow):
        """测试通过索引获取函数"""
        workflow = workflow_from_template

        # 测试获取存在的函数
        fn_0 = workflow[0]
        assert fn_0.fn_id == 0
        assert fn_0.wf_id == workflow.wf_id

        # 测试获取最后一个函数
        last_fn_id = len(workflow) - 1
        last_fn = workflow[last_fn_id]
        assert last_fn.fn_id == last_fn_id

        # 测试获取中间的函数
        if len(workflow) > 2:
            mid_fn_id = len(workflow) // 2
            mid_fn = workflow[mid_fn_id]
            assert mid_fn.fn_id == mid_fn_id

        # 测试索引越界
        with pytest.raises(IndexError):
            _ = workflow[len(workflow)]

    def test_get_predecessor_functions(self, workflow_from_template: Workflow):
        """测试获取前驱函数"""
        workflow = workflow_from_template

        # 测试源点函数（应该没有前驱）
        for source_fn_id in workflow.source_functions:
            predecessors = workflow.get_predecessor_functions(source_fn_id)
            assert len(predecessors) == 0

        # 测试有前驱的函数
        # 找到一个有前驱的函数
        for fn_id in range(len(workflow)):
            predecessors = workflow.get_predecessor_functions(fn_id)
            if len(predecessors) > 0:
                # 验证返回的边的格式
                for edge in predecessors:
                    assert len(edge) == 3
                    source, target, data_size = edge
                    assert target == fn_id
                    assert isinstance(source, int)
                    assert isinstance(data_size, int)
                    assert data_size >= 0
                break

        # 创建一个简单的测试工作流
        simple_workflow = Workflow(
            wf_id=1,
            arrival_time=0.0,
            computations=(1000, 1500, 2000),
            memory_reqs=(512, 768, 1024),
            parallelisms=(2, 3, 4),
            edges=((0, 1, 512), (0, 2, 1024), (1, 2, 256)),
        )

        # 验证前驱关系
        pred_0 = simple_workflow.get_predecessor_functions(0)
        assert len(pred_0) == 0

        pred_1 = simple_workflow.get_predecessor_functions(1)
        assert len(pred_1) == 1
        assert pred_1[0] == (0, 1, 512)

        pred_2 = simple_workflow.get_predecessor_functions(2)
        assert len(pred_2) == 2
        edges_set = {(source, target, size) for source, target, size in pred_2}
        assert (0, 2, 1024) in edges_set
        assert (1, 2, 256) in edges_set
