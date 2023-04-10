from typing import Callable, List, Optional, Union


from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.task_scheduler import PyTaskScheduler,TaskScheduler
from tvm.meta_schedule import _ffi_api
from tvm.meta_schedule.builder import Builder, BuilderResult
from tvm.meta_schedule.cost_model import CostModel
from tvm.meta_schedule.database import Database
from tvm.meta_schedule.logging import get_logger, get_logging_func
from tvm.meta_schedule.measure_callback import MeasureCallback
from tvm.meta_schedule.runner import Runner, RunnerResult
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule.task_scheduler.task_scheduler import TaskRecord

import random
from typing_extensions import Literal
from tvm.runtime import Object
from tvm._ffi import register_object


class AllPyTaskScheduler(TaskScheduler):

    # tasks_: List[TaskRecord]
    # measure_callbacks_: List[MeasureCallback]
    # database_: Optional[Database]
    # cost_model_: Optional[CostModel]
    # remaining_tasks_: int
    # TaskSchedulerType = Union["AllPyTaskScheduler", Literal["gradient", "round-robin"]]

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        next_task_id : int
            The next task id.
        """
        return _ffi_api.TaskSchedulerNextTaskId(self)  # type: ignore # pylint: disable=no-member

    def join_running_task(self, task_id: int) -> List[RunnerResult]:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.

        Returns
        -------
        results : List[RunnerResult]
            The list of results.
        """
        return _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)  # type: ignore # pylint: disable=no-member
    
    def tune(
        self,
        tasks: List[TuneContext],
        task_weights: List[float],
        max_trials_global: int,
        max_trials_per_task: int,
        num_trials_per_iter: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: List[MeasureCallback],
        database: Optional[Database],
        cost_model: Optional[CostModel],
    ) -> None:
        """Auto-tuning.

        Parameters
        ----------
        tasks : List[TuneContext]
            The list of tuning contexts as tasks.
        task_weights : List[float]
            The list of task weights.
        max_trials_global : int
            The maximum number of trials globally.
        max_trials_per_task : int
            The maximum number of trials per task.
        num_trials_per_iter : int
            The number of trials per iteration.
        builder : Builder
            The builder.
        runner : Runner
            The runner.
        measure_callbacks : List[MeasureCallback]
            The list of measure callbacks.
        database : Optional[Database]
            The database.
        cost_model : Optional[CostModel]
            The cost model.
        """
        task_weights = [float(w) for w in task_weights]
        _ffi_api.TaskSchedulerTune(  # type: ignore # pylint: disable=no-member
            self,
            tasks,
            task_weights,
            max_trials_global,
            max_trials_per_task,
            num_trials_per_iter,
            builder,
            runner,
            measure_callbacks,
            database,
            cost_model,
        )

    def python_tune(
        self,
        tasks: List[TuneContext],
        task_weights: List[float],
        max_trials_global: int,
        max_trials_per_task: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: List[MeasureCallback],
        database: Optional[Database],
        cost_model: Optional[CostModel],
    ) -> None:
        task_weights = [float(w) for w in task_weights]
        """
        void TaskSchedulerNode::Tune(Array<TuneContext> ctxs, Array<FloatImm> task_weights,
                             int max_trials_global, int max_trials_per_task,
                             int num_trials_per_iter, Builder builder, Runner runner,
                             Array<MeasureCallback> measure_callbacks, Optional<Database> database,
                             Optional<CostModel> cost_model) {
        CHECK_EQ(ctxs.size(), task_weights.size()) << "ValueError: `task_weights` must have the same "
                                                        "length as `ctxs`";
        int n_tasks = this->remaining_tasks_ = ctxs.size();
        this->measure_callbacks_ = measure_callbacks;
        this->database_ = database;
        this->cost_model_ = cost_model;
        this->tasks_.clear();
        this->tasks_.reserve(n_tasks);
        for (int i = 0; i < n_tasks; ++i) {
            const TuneContext& ctx = ctxs[i];
            double weight = task_weights[i]->value;
            TVM_PY_LOG(INFO, this->logger) << "Initializing Task #" << i << ": " << ctx->task_name;
            TVM_PY_LOG(INFO, ctx->logger) << "Initializing Task #" << i << ": " << ctx->task_name;
            this->tasks_.push_back(TaskRecord(ctx, weight));
            Array<tir::Schedule> design_spaces =
                ctx->space_generator.value()->GenerateDesignSpace(ctx->mod.value());
            TVM_PY_LOG(INFO, ctx->logger) << "Total " << design_spaces.size()
                                        << " design space(s) generated";
            for (int i = 0, n = design_spaces.size(); i < n; ++i) {
            tir::Schedule sch = design_spaces[i];
            tir::Trace trace = sch->trace().value();
            trace = trace->Simplified(true);
            TVM_PY_LOG(INFO, ctx->logger) << "Design space #" << i << ":\n"
                                            << sch->mod() << "\n"
                                            << Concat(trace->AsPython(false), "\n");
            }
            ctx->search_strategy.value()->PreTuning(max_trials_per_task, num_trials_per_iter, design_spaces,
                                                    database, cost_model);
        }

        int num_trials_already = 0;
        for (int task_id; num_trials_already < max_trials_global && (task_id = NextTaskId()) != -1;) {
            TVM_PY_LOG(INFO, this->logger)
                << "TaskScheduler picks Task #" << task_id << ": " << tasks_[task_id]->ctx->task_name;
            TaskRecordNode* task = tasks_[task_id].get();
            ICHECK(!task->is_terminated);
            ICHECK(!task->runner_futures.defined());
            if (static_cast<int>(task->latency_ms.size()) >= max_trials_per_task) {
            TerminateTask(task_id);
            continue;
            }
            if (Optional<Array<MeasureCandidate>> candidates = task->measure_candidates =
                    task->ctx->search_strategy.value()->GenerateMeasureCandidates()) {
            int num_candidates = candidates.value().size();
            num_trials_already += num_candidates;
            TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to builder";
            SendToBuilder(task, builder);
            TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to runner";
            SendToRunner(task, runner);
            } else {
            TerminateTask(task_id);
            }
        }
        for (int task_id = 0; task_id < n_tasks; ++task_id) {
            TaskRecordNode* task = this->tasks_[task_id].get();
            if (!task->is_terminated) {
            if (task->runner_futures.defined()) {
                JoinRunningTask(task_id);
            }
            TerminateTask(task_id);
            }
            task->ctx->search_strategy.value()->PostTuning();
        }
        }

        """
        assert len(tasks) == len(task_weights), "ValueError: `task_weights` must have the same length as `ctxs`"
        n_tasks = len(tasks)
        self.remaining_tasks_ = n_tasks
        self.measure_callbacks_ = measure_callbacks
        self.database_ = database
        self.cost_model_ = cost_model
        self.tasks_ = []



