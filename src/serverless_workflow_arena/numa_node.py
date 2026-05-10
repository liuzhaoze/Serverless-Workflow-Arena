"""numa_node.py

与 NUMA 节点相关的模型
"""

from bisect import bisect_left, bisect_right
from queue import Queue
from typing import Literal, NamedTuple

from .container import Container


class NumaNodeStatus(NamedTuple):
    """NUMA 节点状态

    Attributes:
        cpu (float): CPU 利用率
        memory (float): 内存利用率
        load (float): 负载水平
        total_parallelism (int): NUMA 节点上运行的总并行度
        free_memory (int): 可用内存大小 (MB)
        time_to_first_completion (float): 第一个容器执行完成需要经历的时间 (单位：秒)
        time_to_last_completion (float): 最后一个容器执行完成需要经历的时间 (单位：秒)
    """

    cpu: float
    memory: float
    load: float
    total_parallelism: int
    free_memory: int
    time_to_first_completion: float
    time_to_last_completion: float


class NumaNode:
    """NUMA 节点模型

    Args:
        node_id (int): NUMA节点ID
        cpu (int): NUMA节点的 CPU 数量
        memory (int): NUMA节点的内存大小 (MB)
        single_core_speed (int): 单核计算速度 (单个 CPU 核心每秒可执行的计算操作数量)
    """

    __slots__ = (
        "node_id",
        "cpu",
        "memory",
        "single_core_speed",
        "computing_capacity",
        "_waiting_containers",
        "_running_containers",
        "_current_time",
        "_rum",
        "_esm",
    )

    def __init__(self, node_id: int, cpu: int, memory: int, single_core_speed: int):
        self.node_id: int = node_id
        self.cpu: int = cpu
        self.memory: int = memory
        self.single_core_speed: int = single_core_speed
        self.computing_capacity: int = cpu * single_core_speed

        self._waiting_containers: Queue[Container] = Queue()
        self._running_containers: list[Container] = []
        self._current_time: float = 0.0
        self._rum = ResourceUtilizationManager()
        self._esm = ExecutionSpeedManager()

    def reset(self):
        """重置 NUMA 节点的状态"""
        self._waiting_containers = Queue()
        self._running_containers.clear()
        self._current_time = 0.0
        self._rum.reset(0.0, 0.0, 0.0, 0.0, 0, self.memory)
        self._esm.reset()

    def on_server_startup(self, time: float, cold_start_latency: float):
        """当服务器启动时，需要执行的操作

        Args:
            time (float): 服务器启动时间
            cold_start_latency (float): 服务器冷启动时间
        """

        if time < self._current_time:
            raise ValueError(f"Server startup time {time} cannot be earlier than current time {self._current_time}")

        if self._running_containers or not self._waiting_containers.empty():
            raise RuntimeError("Cannot start a server that is already running")

        # 更新当前时间到服务器启动完毕
        self._current_time = time + cold_start_latency

    def on_container_creation(self, c: Container):
        """当一个容器被创建时，需要执行的操作

        Args:
            c (Container): 新容器实例
        """

        # 计算容器的创建时间
        # 理论上讲，容器创建之前需要经历服务器冷启动和数据传输两个阶段，但此处计算创建时间时并没有体现冷启动时间
        # 这是因为 Server.startup_at 方法在处理冷启动时会调用 NumaNode.on_server_startup 方法，将服务器的所有 NUMA 节点的当前时间更新到服务器启动完成后的时间
        # 这样能够保证所有 NUMA 节点都受到了服务器冷启动的影响
        # 之所以不使用 creation_time = submission_time + cold_start_time + data_transfer_time 来计算容器创建时间
        # 是因为后续提交到该服务器的函数虽然不会触发冷启动（即 cold_start_time = 0），但是那些提交时间早于服务器启动完成的函数的容器也应该受到冷启动的影响，在服务器启动之后创建
        # 而上述计算方法可能会出现 creation_time = submission_time + 0 + data_transfer_time 小于服务器启动完成时间的情况，没有体现冷启动对后续提交到该服务器上的函数的容器的创建时间产生的影响
        # 因此，只有在服务器冷启动时调用 on_server_startup 方法，将属于该服务器的 NUMA 节点的当前时间更新到服务器启动完成的时间，并使用下面的方法计算容器创建时间
        # 才能够确保后续提交到该服务器的函数的容器创建都能受到冷启动的影响，保证容器创建时间不会早于服务器启动完成后的时间
        creation_time = max(self._current_time, c.submission_time) + c.data_transfer_time
        # 不用担心在 submit_time 和 creation_time 之间出现其他函数完成导致这些函数的后继函数比当前函数早提交的情况
        # 因为那些后继函数的提交时间一定比当前函数的提交时间晚，先处理的必然是当前函数
        c.create(creation_time)

        r = self._rum.get_record_at(creation_time)

        # 如果在容器运行提交时，NUMA 节点的剩余内存小于容器所需内存，则将容器放入等待队列
        if r.free_memory < c.memory_alloc:
            self._waiting_containers.put(c)
            return

        # 如果 NUMA 节点的剩余内存足够，则直接运行容器
        c.run(creation_time)
        self._running_containers.append(c)

        # 更新新容器运行后所有容器的完成时间
        self._update_completion_times()

    def on_container_completion(self) -> Container:
        """当一个容器完成时，需要执行的操作

        Returns:
            Container: 已完成的容器实例
        """

        # 将最早完成的容器从运行列表中移除
        index, completion_time = self.get_earliest_finished()
        if index is None:
            raise RuntimeError("No running containers to complete")

        container = self._running_containers.pop(index)

        # 更新该容器完成时其他所有运行容器的剩余计算量
        self._update_remaining_computations(completion_time)

        # 更新当前时间到该容器完成时间
        self._current_time = completion_time

        # 更新该容器完成后所有容器的完成时间
        self._update_completion_times()

        # 容器完成后会释放内存资源，尝试从等待队列中取出容器运行
        self._run_waiting_containers(self._current_time)

        return container

    def _update_completion_times(self):
        """更新所有正在运行的容器的完成时间

        同时更新:

        - 资源利用率记录
        - 每个容器的执行速度记录
        """

        # 初始化已经过去了的时间
        elapsed_time = self._current_time

        # 清除当前时间之后的资源利用率记录
        self._rum.clear_after(elapsed_time)

        # 清除所有函数的执行速度记录
        self._esm.reset()

        # 初始化活动容器及其剩余计算量 (字典：容器索引 -> 剩余计算量)
        active_containers = {
            i: c.remaining_computation
            for i, c in enumerate(self._running_containers)
            if c.remaining_computation > 0 and c.start_time <= elapsed_time
        }

        # 初始化未来要启动的容器（元组列表：(容器索引, 启动时间)）
        future_containers = [
            (i, c.start_time)
            for i, c in enumerate(self._running_containers)
            if c.remaining_computation > 0 and c.start_time > elapsed_time
        ]
        # 初始化未来要启动的容器的执行速度记录
        for i, _ in future_containers:
            # 如果不为 future_containers 初始化执行速度记录，那么根据下面循环中的逻辑，这些容器的第一条执行速度记录的时间戳为 start_time (start_time > current_time)
            # 这就会导致：如果需要计算从 current_time 到 start_time 之前的某一时刻已完成的计算量时，ExecutionSpeedManager.get_computation_until 方法会抛出 ValueError 异常（因为没有 start_time 之前的记录）
            # 而 active_containers 中的容器则不会有这个问题，因为它们在 current_time 时就已经开始运行 (start_time <= current_time)，所以第一条执行速度记录的时间戳一定是 current_time
            self._esm.add_record(
                self._running_containers[i].wf_id,
                self._running_containers[i].fn_id,
                SpeedRecord(timestamp=elapsed_time, speed=0),
            )
        # 将未来容器按启动时间排序
        future_containers.sort(key=lambda x: x[1])

        # 初始化总并行度和已使用内存
        total_parallelism = sum(self._running_containers[i].parallelism for i in active_containers.keys())
        used_memory = sum(self._running_containers[i].memory_alloc for i in active_containers.keys())

        # 迭代处理容器执行结束和容器开始执行事件，直到所有容器都完成
        while True:
            # 计算单个并行度的执行速度
            if total_parallelism <= self.cpu:
                single_parallelism_speed = self.single_core_speed
            else:
                single_parallelism_speed = self.computing_capacity // total_parallelism

            # 计算活动容器的执行速度和完成时间
            actual_speeds: dict[int, int] = {}  # 容器索引 -> 实际执行速度
            total_actual_speed = 0
            for i in active_containers.keys():
                memory_bottleneck = min(
                    1.0, self._running_containers[i].memory_alloc / self._running_containers[i].memory_req
                )
                actual_speed = int(
                    memory_bottleneck * single_parallelism_speed * self._running_containers[i].parallelism
                )
                actual_speeds[i] = actual_speed
                total_actual_speed += actual_speed
                self._esm.add_record(
                    self._running_containers[i].wf_id,
                    self._running_containers[i].fn_id,
                    SpeedRecord(timestamp=elapsed_time, speed=actual_speed),
                )
                self._running_containers[i].finish(elapsed_time + active_containers[i] / actual_speed)

            # 记录资源利用率
            self._rum.add_record(
                UtilizationRecord(
                    timestamp=elapsed_time,
                    cpu=total_actual_speed / self.computing_capacity,
                    memory=used_memory / self.memory,
                    load=total_parallelism / self.cpu,
                    total_parallelism=total_parallelism,
                    free_memory=self.memory - used_memory,
                )
            )

            # 找到下一个最早发生的事件
            next_event_time: float = float("inf")
            next_event_type: Literal["start", "finish"] | None = None
            next_event_container: int | None = None

            # 选择最先发生的容器开始事件
            if future_containers:
                next_event_container, next_event_time = future_containers[0]
                next_event_type = "start"

            # 检查活动容器的完成事件是否有更早的
            for i in active_containers.keys():
                if self._running_containers[i].completion_time < next_event_time:
                    next_event_time = self._running_containers[i].completion_time
                    next_event_type = "finish"
                    next_event_container = i

            # 如果没有事件，则退出循环
            if next_event_container is None:
                break

            # 更新下一个事件发生时，所有活动容器的剩余计算量
            time_delta = next_event_time - elapsed_time
            for i in active_containers.keys():
                active_containers[i] -= int(actual_speeds[i] * time_delta)

            # 更新已经过去的时间
            elapsed_time = next_event_time

            if next_event_type == "finish":
                # 容器完成事件：从活动容器中移除
                del active_containers[next_event_container]
                total_parallelism -= self._running_containers[next_event_container].parallelism
                used_memory -= self._running_containers[next_event_container].memory_alloc

            elif next_event_type == "start":
                # 容器开始事件：从未来容器列表中移除，并加入活动容器
                future_containers.pop(0)
                active_containers[next_event_container] = self._running_containers[
                    next_event_container
                ].remaining_computation
                total_parallelism += self._running_containers[next_event_container].parallelism
                used_memory += self._running_containers[next_event_container].memory_alloc

            # 清除新的 elapsed_time 及之后的资源利用率记录和执行速度记录，以便于下一轮记录新的数据
            # 这一步主要为了防止如下问题：
            # 当 future_containers 中存在 start_time 相同的容器时，
            # 迭代第一个容器时，会在资源利用率记录和执行速度记录中添加 start_time 时刻的记录；
            # 迭代第二个容器时，也会添加 start_time 时刻的记录，并且这个记录比前一个记录新（其代表两个容器同时运行时的状态）。
            # 因此需要清除前一个记录，在保留最新的记录的同时，保证每个时间点只有一条记录。
            self._rum.clear_after(elapsed_time)
            for i in active_containers.keys():
                self._esm.clear_after(
                    self._running_containers[i].wf_id,
                    self._running_containers[i].fn_id,
                    elapsed_time,
                )

            # 继续下一轮迭代，重新计算剩余容器的执行速度和完成时间

    def _update_remaining_computations(self, time: float):
        """更新所有正在运行的容器到指定时间点的剩余计算量

        Args:
            time (float): 时间点
        """
        if time < self._current_time:
            raise ValueError(f"Update time {time} cannot be earlier than current time {self._current_time}")

        for c in self._running_containers:
            if c.remaining_computation <= 0:
                # _update_completion_times 方法在初始化 active_containers 和 future_containers 时会排除 remaining_computation <= 0 的容器
                # 因此 remaining_computation <= 0 的容器没有执行速度记录，更新它们的剩余计算量会抛出 KeyError 异常
                continue
            completed_computation = self._esm.get_computation_until(c.wf_id, c.fn_id, time)
            c.remaining_computation -= completed_computation

    def _run_waiting_containers(self, time: float):
        """不断地从等待队列中取出容器运行，直到内存不足以运行下一个容器为止

        Args:
            time (float): 容器完成事件的时间
        """

        while not self._waiting_containers.empty():
            with self._waiting_containers.mutex:
                c: Container = self._waiting_containers.queue[0]

            # 根据 on_container_creation 方法中的注释对容器创建时间计算方法的说明
            # 容器的创建时间可能会因为冷启动和数据传输晚于当前容器完成事件的时间
            # 所以容器的实际启动时间是两者中较晚的时间
            start_time = max(time, c.creation_time)

            # 检查容器启动时内存是否足够，如果不够则停止尝试启动容器
            r = self._rum.get_record_at(start_time)
            if r.free_memory < c.memory_alloc:
                break

            # 剩余内存足够，取出容器运行
            container = self._waiting_containers.get()
            container.run(start_time)
            self._running_containers.append(container)

            # 新容器运行后更新所有容器的完成时间，同时也会更新剩余内存记录
            self._update_completion_times()

    def get_earliest_finished(self) -> tuple[int | None, float]:
        """获得最早完成的容器的索引和完成时间

        Returns:
            (int | None, float): 最早完成的容器的索引和完成时间；如果没有运行中的容器，返回 `(None, float("inf"))`
        """

        if not self._running_containers:
            if not self._waiting_containers.empty():
                # 理论上不可能存在 NUMA 节点没有运行中的容器但有等待中的容器的情况
                # 因为 self.on_container_completion 方法会保证只要有容器完成（包括最后一个容器完成）
                # 就会尝试从等待队列中取出容器运行，直到内存不足或等待队列为空
                raise RuntimeError("No running containers, but there are waiting containers")
            return (None, float("inf"))

        return min(((i, c.completion_time) for i, c in enumerate(self._running_containers)), key=lambda x: x[1])

    def get_latest_finished(self) -> tuple[int | None, float]:
        """获得最晚完成的容器的索引和完成时间

        Returns:
            (int | None, float): 最晚完成的容器的索引和完成时间；如果没有运行中的容器，返回 `(None, float("-inf"))`
        """

        if not self._running_containers:
            if not self._waiting_containers.empty():
                # 理论上不可能存在 NUMA 节点没有运行中的容器但有等待中的容器的情况
                # 因为 self.on_container_completion 方法会保证只要有容器完成（包括最后一个容器完成）
                # 就会尝试从等待队列中取出容器运行，直到内存不足或等待队列为空
                raise RuntimeError("No running containers, but there are waiting containers")
            return (None, float("-inf"))

        return max(((i, c.completion_time) for i, c in enumerate(self._running_containers)), key=lambda x: x[1])

    def get_status_at(self, time: float) -> NumaNodeStatus:
        """获得指定时间点的 NUMA 节点状态

        Args:
            time (float): 时间点

        Returns:
            NumaNodeStatus: NUMA 节点状态
        """
        r = self._rum.get_record_at(time)

        c, t = self.get_earliest_finished()
        time_to_first_completion = 0.0 if c is None else max(0.0, t - time)

        c, t = self.get_latest_finished()
        time_to_last_completion = 0.0 if c is None else max(0.0, t - time)

        return NumaNodeStatus(
            r.cpu,
            r.memory,
            r.load,
            r.total_parallelism,
            r.free_memory,
            time_to_first_completion,
            time_to_last_completion,
        )


class UtilizationRecord(NamedTuple):
    """NUMA 节点的资源利用率记录

    Attributes:
        timestamp (float): 时间戳
        cpu (float): CPU 利用率
        memory (float): 内存利用率
        load (float): 负载水平
        total_parallelism (int): NUMA 节点上运行的总并行度
        free_memory (int): 可用内存大小 (MB)
    """

    timestamp: float
    cpu: float
    memory: float
    load: float
    total_parallelism: int
    free_memory: int


class ResourceUtilizationManager:
    """记录 NUMA 节点的资源利用率

    每条记录表示自该时间点起，直到下一条记录时间点为止的资源利用率情况。
    """

    __slots__ = ("records",)

    def __init__(self):
        self.records: list[UtilizationRecord] = []  # 元素必须保证按 timestamp 升序

    def reset(self, timestamp: float, cpu: float, memory: float, load: float, total_parallelism: int, free_memory: int):
        """删除所有记录，并添加一条表示初始状态的记录

        Args:
            timestamp (float): 时间戳
            cpu (float): CPU 利用率
            memory (float): 内存利用率
            load (float): 负载水平
            total_parallelism (int): NUMA 节点上运行的总并行度
            free_memory (int): 可用内存大小 (MB)
        """
        self.records = [UtilizationRecord(timestamp, cpu, memory, load, total_parallelism, free_memory)]

    def clear_after(self, time: float):
        """删除指定时间点以及之后的所有记录

        Args:
            time (float): 时间点
        """

        # 使用二分查找找到第一个大于等于 time 的记录索引
        i = bisect_left(self.records, time, key=lambda r: r.timestamp)

        if i == len(self.records):
            # 没有需要删除的记录直接返回，避免不必要的切片操作导致的内存分配、副本创建和垃圾回收开销
            return

        self.records = self.records[:i]

    def add_record(self, record: UtilizationRecord):
        """添加资源利用率记录

        Args:
            record (UtilizationRecord): 资源利用率记录
        """

        if self.records and record.timestamp <= self.records[-1].timestamp:
            raise ValueError("New record timestamp must be greater than the last record timestamp")

        self.records.append(record)

    def get_record_at(self, time: float) -> UtilizationRecord:
        """获得指定时间点的资源利用率

        Args:
            time (float): 时间点

        Returns:
            UtilizationRecord: 当前时间点的资源利用率
        """

        # 使用二分查找找到最后一个小于等于 time 的记录索引
        i = bisect_right(self.records, time, key=lambda r: r.timestamp) - 1

        if i < 0:
            raise ValueError(
                f"No utilization record available before the specified time; the earliest record is at {self.records[0].timestamp}, requested time is {time}"
            )

        return self.records[i]


class SpeedRecord(NamedTuple):
    """函数的执行速度记录

    Attributes:
        timestamp (float): 时间戳
        speed (int): 执行速度 (每秒可执行的计算操作数量)
    """

    timestamp: float
    speed: int


class ExecutionSpeedManager:
    """记录 NUMA 节点上每个容器的执行速度变化情况

    底层结构为一个字典，字典的键为工作流 ID 和函数 ID 的元组，值为该函数的执行速度记录列表。

    记录列表中的每条记录表示自该时间点起，直到下一条记录时间点为止的执行速度。
    """

    __slots__ = ("records",)

    def __init__(self):
        self.records: dict[tuple[int, int], list[SpeedRecord]] = {}

    def reset(self):
        """删除所有函数的执行速度记录"""
        self.records.clear()

    def clear_after(self, wf_id: int, fn_id: int, time: float):
        """删除指定函数在指定时间点以及之后的所有执行速度记录

        Args:
            wf_id (int): 工作流ID
            fn_id (int): 函数ID
            time (float): 时间点
        """

        key = (wf_id, fn_id)
        if key not in self.records:
            return

        records = self.records[key]

        # 使用二分查找找到第一个大于等于 time 的记录索引
        i = bisect_left(records, time, key=lambda r: r.timestamp)

        if i == len(records):
            # 没有需要删除的记录直接返回，避免不必要的切片操作导致的内存分配、副本创建和垃圾回收开销
            return

        self.records[key] = records[:i]

    def add_record(self, wf_id: int, fn_id: int, record: SpeedRecord):
        """添加函数的执行速度记录

        Args:
            wf_id (int): 工作流ID
            fn_id (int): 函数ID
            record (SpeedRecord): 执行速度记录
        """

        key = (wf_id, fn_id)
        if key not in self.records:
            self.records[key] = []

        if self.records[key] and record.timestamp <= self.records[key][-1].timestamp:
            raise ValueError("New speed record timestamp must be greater than the last record timestamp")

        self.records[key].append(record)

    # 因为 `ExecutionSpeedManager` 的 `reset` 和 `add_record` 方法只在 `NumaNode._update_completion_times` 中调用，
    # 它们的调用逻辑保证了第一条记录的时间戳一定是 `NumaNode._current_time`。
    # （只有在 `NumaNode._update_completion_times` 中初始化 future_containers 的执行速度记录才能保证上述断言成立；否则这些容器的第一条记录的时间戳是它们各自的 start_time，比 `NumaNode._current_time` 要大。）
    # 而 `ExecutionSpeedManager` 的 `get_computation_until` 方法只在 `NumaNode._update_remaining_computations` 中调用，
    # 其目的是计算从 `NumaNode._current_time` 开始到指定时间点为止，正在运行的容器已完成的计算量，从而更新容器的剩余计算量。
    # 综上，`ExecutionSpeedManager.get_computation_until` 方法只需要传入截止时间点即可。
    # 这是因为该方法将第一条记录的时间戳作为起始时间点，始终等于 `NumaNode._current_time`。

    def get_computation_until(self, wf_id: int, fn_id: int, time: float) -> int:
        """获得指定函数到指定时间点已完成的计算量

        Args:
            wf_id (int): 工作流ID
            fn_id (int): 函数ID
            time (float): 时间点

        Returns:
            int: 到指定时间点函数已完成的计算量
        """

        completed_computation = 0
        records = self.records[(wf_id, fn_id)]

        if time < records[0].timestamp:
            raise ValueError(
                f"No speed record available before the specified time; the earliest record is at {records[0].timestamp}, requested time is {time}"
            )

        # 遍历执行速度记录，计算每段时间内完成的计算量
        for i, r in enumerate(records):
            # 确定时间段
            period_start = r.timestamp
            if i < len(records) - 1:
                # 当前记录不是最后一条记录，时间段持续到下一条记录开始或指定时间
                period_end = min(records[i + 1].timestamp, time)
            else:
                # 当前记录是最后一条记录，时间段到指定时间
                period_end = time

            # 计算这段时间内完成的计算量
            assert period_end >= period_start
            completed_computation += int(r.speed * (period_end - period_start))

            # 如果时间段已覆盖指定时间点，则停止计算
            if period_end >= time:
                break

        return completed_computation
