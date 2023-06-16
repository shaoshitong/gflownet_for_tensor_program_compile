# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Gradient Based Task Scheduler"""
from typing import Callable, List, Optional, Union

from typing_extensions import Literal

from tvm._ffi import register_object

from .. import _ffi_api
from ..builder import Builder, BuilderResult
from ..cost_model import CostModel
from ..database import Database
from ..logging import get_logger, get_logging_func
from ..measure_callback import MeasureCallback
from ..runner import Runner, RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext
from ..utils import derived_object
from .task_scheduler import TaskScheduler

logger = get_logger(__name__)  # pylint: disable=invalid-name


@register_object("meta_schedule.AllPythonBased")
class AllPythonBased(TaskScheduler):
    """AllPython Based Task Scheduler"""

    def __init__(
        self,
        *,
        alpha: float = 0.2,
        window_size: int = 3,
        seed: int = -1,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        alpha : float = 0.2
            The parameter alpha in gradient computation.
        window_size : int = 3
            The parameter to control backward window size in gradient computation.
        seed : int = -1
            The random seed.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerAllPythonBased,  # type: ignore # pylint: disable=no-member
            get_logging_func(logger),
            alpha,
            window_size,
            seed,
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
