"""
数据处理模块
定义FJSP问题的数据结构和初始化方法
"""

class Job:
    """工件类"""
    def __init__(self, job_id):
        self.id = job_id
        self.operations = []
    
    def add_operation(self, operation):
        """添加工序"""
        self.operations.append(operation)
    
    def get_operation_count(self):
        """获取工序数量"""
        return len(self.operations)

class Operation:
    """工序类"""
    def __init__(self, job_id, op_id):
        self.job_id = job_id
        self.id = op_id
        self.machines = {}  # 机器ID: 加工时间
    
    def add_machine(self, machine_id, processing_time):
        """添加可选机器及加工时间"""
        self.machines[machine_id] = processing_time

class FJSPProblem:
    """柔性作业车间调度问题类"""
    def __init__(self):
        self.jobs = []
        self.machines = 6  # 6台机器
        self.num_jobs = 10  # 10个工件
        self._initialize_problem()
    
    def _initialize_problem(self):
        """初始化问题数据"""
        # 工件1 (6个工序)
        job1 = Job(1)
        # 工序1
        op1 = Operation(1, 1)
        op1.add_machine(1, 5)
        op1.add_machine(3, 4)
        job1.add_operation(op1)
        # 工序2
        op2 = Operation(1, 2)
        op2.add_machine(5, 3)
        op2.add_machine(3, 5)
        op2.add_machine(2, 1)
        job1.add_operation(op2)
        # 工序3
        op3 = Operation(1, 3)
        op3.add_machine(3, 4)
        op3.add_machine(6, 2)
        job1.add_operation(op3)
        # 工序4
        op4 = Operation(1, 4)
        op4.add_machine(6, 5)
        op4.add_machine(2, 6)
        op4.add_machine(1, 1)
        job1.add_operation(op4)
        # 工序5
        op5 = Operation(1, 5)
        op5.add_machine(3, 1)
        job1.add_operation(op5)
        # 工序6
        op6 = Operation(1, 6)
        op6.add_machine(6, 6)
        op6.add_machine(3, 6)
        op6.add_machine(4, 3)
        job1.add_operation(op6)
        self.jobs.append(job1)
        
        # 工件2 (5个工序)
        job2 = Job(2)
        # 工序1
        op1 = Operation(2, 1)
        op1.add_machine(2, 6)
        job2.add_operation(op1)
        # 工序2
        op2 = Operation(2, 2)
        op2.add_machine(3, 1)
        job2.add_operation(op2)
        # 工序3
        op3 = Operation(2, 3)
        op3.add_machine(1, 2)
        job2.add_operation(op3)
        # 工序4
        op4 = Operation(2, 4)
        op4.add_machine(2, 6)
        op4.add_machine(4, 6)
        job2.add_operation(op4)
        # 工序5
        op5 = Operation(2, 5)
        op5.add_machine(6, 5)
        op5.add_machine(2, 6)
        op5.add_machine(1, 1)
        job2.add_operation(op5)
        self.jobs.append(job2)
        
        # 工件3 (5个工序)
        job3 = Job(3)
        # 工序1
        op1 = Operation(3, 1)
        op1.add_machine(2, 6)
        job3.add_operation(op1)
        # 工序2
        op2 = Operation(3, 2)
        op2.add_machine(3, 4)
        op2.add_machine(6, 2)
        job3.add_operation(op2)
        # 工序3
        op3 = Operation(3, 3)
        op3.add_machine(6, 5)
        op3.add_machine(2, 6)
        op3.add_machine(1, 1)
        job3.add_operation(op3)
        # 工序4
        op4 = Operation(3, 4)
        op4.add_machine(3, 4)
        op4.add_machine(2, 6)
        op4.add_machine(6, 6)
        job3.add_operation(op4)
        # 工序5
        op5 = Operation(3, 5)
        op5.add_machine(1, 1)
        op5.add_machine(5, 5)
        job3.add_operation(op5)
        self.jobs.append(job3)
        
        # 工件4 (5个工序)
        job4 = Job(4)
        # 工序1
        op1 = Operation(4, 1)
        op1.add_machine(6, 5)
        op1.add_machine(2, 6)
        op1.add_machine(1, 1)
        job4.add_operation(op1)
        # 工序2
        op2 = Operation(4, 2)
        op2.add_machine(2, 6)
        job4.add_operation(op2)
        # 工序3
        op3 = Operation(4, 3)
        op3.add_machine(3, 1)
        job4.add_operation(op3)
        # 工序4
        op4 = Operation(4, 4)
        op4.add_machine(5, 3)
        op4.add_machine(3, 5)
        op4.add_machine(2, 1)
        job4.add_operation(op4)
        # 工序5
        op5 = Operation(4, 5)
        op5.add_machine(3, 4)
        op5.add_machine(6, 2)
        job4.add_operation(op5)
        self.jobs.append(job4)
        
        # 工件5 (6个工序)
        job5 = Job(5)
        # 工序1
        op1 = Operation(5, 1)
        op1.add_machine(5, 3)
        op1.add_machine(3, 5)
        op1.add_machine(2, 1)
        job5.add_operation(op1)
        # 工序2
        op2 = Operation(5, 2)
        op2.add_machine(6, 5)
        op2.add_machine(2, 6)
        op2.add_machine(1, 1)
        job5.add_operation(op2)
        # 工序3
        op3 = Operation(5, 3)
        op3.add_machine(2, 6)
        job5.add_operation(op3)
        # 工序4
        op4 = Operation(5, 4)
        op4.add_machine(1, 5)
        op4.add_machine(3, 4)
        job5.add_operation(op4)
        # 工序5
        op5 = Operation(5, 5)
        op5.add_machine(2, 6)
        op5.add_machine(4, 6)
        job5.add_operation(op5)
        # 工序6
        op6 = Operation(5, 6)
        op6.add_machine(3, 4)
        op6.add_machine(2, 6)
        op6.add_machine(6, 6)
        job5.add_operation(op6)
        self.jobs.append(job5)
        
        # 工件6 (6个工序)
        job6 = Job(6)
        # 工序1
        op1 = Operation(6, 1)
        op1.add_machine(3, 4)
        op1.add_machine(6, 2)
        job6.add_operation(op1)
        # 工序2
        op2 = Operation(6, 2)
        op2.add_machine(1, 2)
        job6.add_operation(op2)
        # 工序3
        op3 = Operation(6, 3)
        op3.add_machine(3, 4)
        op3.add_machine(2, 6)
        op3.add_machine(6, 6)
        job6.add_operation(op3)
        # 工序4
        op4 = Operation(6, 4)
        op4.add_machine(2, 6)
        job6.add_operation(op4)
        # 工序5
        op5 = Operation(6, 5)
        op5.add_machine(6, 5)
        op5.add_machine(2, 6)
        op5.add_machine(1, 1)
        job6.add_operation(op5)
        # 工序6
        op6 = Operation(6, 6)
        op6.add_machine(1, 3)
        op6.add_machine(4, 2)
        job6.add_operation(op6)
        self.jobs.append(job6)
        
        # 工件7 (5个工序)
        job7 = Job(7)
        # 工序1
        op1 = Operation(7, 1)
        op1.add_machine(6, 1)
        job7.add_operation(op1)
        # 工序2
        op2 = Operation(7, 2)
        op2.add_machine(1, 3)
        op2.add_machine(4, 2)
        job7.add_operation(op2)
        # 工序3
        op3 = Operation(7, 3)
        op3.add_machine(3, 4)
        op3.add_machine(2, 6)
        op3.add_machine(6, 6)
        job7.add_operation(op3)
        # 工序4
        op4 = Operation(7, 4)
        op4.add_machine(2, 6)
        op4.add_machine(5, 1)
        op4.add_machine(1, 6)
        job7.add_operation(op4)
        # 工序5
        op5 = Operation(7, 5)
        op5.add_machine(3, 1)
        job7.add_operation(op5)
        self.jobs.append(job7)
        
        # 工件8 (5个工序)
        job8 = Job(8)
        # 工序1
        op1 = Operation(8, 1)
        op1.add_machine(3, 4)
        op1.add_machine(6, 2)
        job8.add_operation(op1)
        # 工序2
        op2 = Operation(8, 2)
        op2.add_machine(3, 4)
        op2.add_machine(2, 6)
        op2.add_machine(6, 6)
        job8.add_operation(op2)
        # 工序3
        op3 = Operation(8, 3)
        op3.add_machine(6, 5)
        op3.add_machine(2, 6)
        op3.add_machine(1, 1)
        job8.add_operation(op3)
        # 工序4
        op4 = Operation(8, 4)
        op4.add_machine(2, 6)
        job8.add_operation(op4)
        # 工序5
        op5 = Operation(8, 5)
        op5.add_machine(2, 6)
        op5.add_machine(4, 6)
        job8.add_operation(op5)
        self.jobs.append(job8)
        
        # 工件9 (6个工序)
        job9 = Job(9)
        # 工序1
        op1 = Operation(9, 1)
        op1.add_machine(6, 1)
        job9.add_operation(op1)
        # 工序2
        op2 = Operation(9, 2)
        op2.add_machine(1, 1)
        op2.add_machine(5, 5)
        job9.add_operation(op2)
        # 工序3
        op3 = Operation(9, 3)
        op3.add_machine(6, 6)
        op3.add_machine(3, 6)
        op3.add_machine(4, 3)
        job9.add_operation(op3)
        # 工序4
        op4 = Operation(9, 4)
        op4.add_machine(1, 2)
        job9.add_operation(op4)
        # 工序5
        op5 = Operation(9, 5)
        op5.add_machine(3, 4)
        op5.add_machine(2, 6)
        op5.add_machine(6, 6)
        job9.add_operation(op5)
        # 工序6
        op6 = Operation(9, 6)
        op6.add_machine(2, 6)
        op6.add_machine(4, 6)
        job9.add_operation(op6)
        self.jobs.append(job9)
        
        # 工件10 (6个工序)
        job10 = Job(10)
        # 工序1
        op1 = Operation(10, 1)
        op1.add_machine(3, 4)
        op1.add_machine(6, 2)
        job10.add_operation(op1)
        # 工序2
        op2 = Operation(10, 2)
        op2.add_machine(3, 4)
        op2.add_machine(2, 6)
        op2.add_machine(6, 6)
        job10.add_operation(op2)
        # 工序3
        op3 = Operation(10, 3)
        op3.add_machine(5, 3)
        op3.add_machine(3, 5)
        op3.add_machine(2, 1)
        job10.add_operation(op3)
        # 工序4
        op4 = Operation(10, 4)
        op4.add_machine(6, 1)
        job10.add_operation(op4)
        # 工序5
        op5 = Operation(10, 5)
        op5.add_machine(2, 6)
        op5.add_machine(4, 6)
        job10.add_operation(op5)
        # 工序6
        op6 = Operation(10, 6)
        op6.add_machine(1, 3)
        op6.add_machine(4, 2)
        job10.add_operation(op6)
        self.jobs.append(job10)
    
    def get_total_operations(self):
        """获取总工序数"""
        return sum(job.get_operation_count() for job in self.jobs)