
import tvm
from typing import TYPE_CHECKING, Callable, List, Optional, Union

# isort: off
from typing_extensions import Literal

# isort: on
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule,Trace
import tvm
from .. import _ffi_api
from ..arg_info import ArgInfo
from ..runner import RunnerResult
from ..utils import cpu_count, derived_object, get_global_func_with_default_on_worker
from ..cost_model import CostModel
from ..database import Database
from ..tune_context import TuneContext
from .search_strategy import PySearchStrategy,SearchStrategy,MeasureCandidate
from ..profiler import Profiler
import copy
from multiprocessing import Value

def forkseed(rand_state):
    rand_state = int(rand_state)
    return (rand_state * 32767) % 1999999973

class Item:
    def __init__(self,postproc) -> None:
        self.postproc = postproc
        self.fail_counter = Value('i', 0)

class ThreadedTraceApply:
    def __init__(self,postprocs) -> None:
        self.n_ = len(postprocs)
        self.items_ = [Item(postprocs[i]) for i in range(self.n_)]

    def Apply(self,mod,trace,rand_state):
        sch = Schedule.Traced(mod,forkseed(rand_state),0,tir.ScheduleErrorRenderLevel.kNone)
        """
      Optional<tir::Schedule> Apply(const IRModule& mod, const tir::Trace& trace,
                                TRandState* rand_state) {
    tir::Schedule sch =
        tir::Schedule::Traced(mod,
                              /*rand_state=*/ForkSeed(rand_state),
                              /*debug_mode=*/0,
                              /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);

    trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
    sch->EnterPostproc();

    for (int i = 0; i < n_; ++i) {
      Item& item = items_[i];
      if (!item.postproc->Apply(sch)) {
        item.fail_counter++;
        return NullOpt;
      }
    }
    return sch;
  }
        """
    
@register_object("meta_schedule.State")
class State:
    """
    The state of the search strategy, used for decoupling the evolutionary search.
    Parameters
    ----------
    self: EvolutionarySearch
        The search strategy itself.
    max_trials: int
        The number of total trials.
    num_trials_per_iter: int
        The number of trials per iteration.
    st: int
        `[st, ed)` are the indices of the next batch of candidates. 
    ed: int
        `[st, ed)` are the indices of the next batch of candidates.
    num_empty_iters: int
        The counter of returning empty results.
    design_spaces: list[tir.Trace]
        The design spaces. Decisions are not used so traces only.
    per_thread_data_: list[PerThreadData]
        Pre thread data including module to be tuned and random state.
    measured_workloads_: IRModuleSet
        The workloads that are already measured.
    database_: Database
        A Database for selecting useful candidates.
    cost_model_: CostModel  
        A cost model helping to explore the search space
    token_: Workload
        The token registered for the given workload in database.
    """
    def __init__(self, context, searchstrategy, max_trials, num_trials_per_iter, design_space_schedules, database, cost_model) -> None:
        self.context = context
        self.sraechstrategy = searchstrategy
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_space_schedules = design_space_schedules
        self.database_ = database
        self.cost_model_ = cost_model
        self.st = 0
        self.num_empty_iters = 0
        self.measured_workloads_ = None
        self.design_spaces = []
        for space in self.design_space_schedules:
            self.design_spaces.append(space.trace.trace.simplified(True))
        self.mod = context.mod
        self.sraechstrategy.per_thread_data_ = []
        for i in range(self.context.num_threads):
            self.sraechstrategy.per_thread_data_[i].mod = copy.deepcopy(self.mod)
            self.sraechstrategy.per_thread_data_[i].rand_state = forkseed(self.sraechstrategy.rand_state_)
        self.token_ = database.commit_workload(self.mod)

    def pickbestfromdatabase(self,num) -> List[Schedule]:
        """
        std::vector<Schedule> EvolutionarySearchNode::State::PickBestFromDatabase(int num) {
  auto _ = Profiler::TimedScope("EvoSearch/PickBestFromDatabase");
  std::vector<tir::Trace> measured_traces;
  measured_traces.reserve(num);
  Array<TuningRecord> top_records = this->database_->GetTopK(this->token_, num);
  for (TuningRecord record : top_records) {
    measured_traces.push_back(record->trace);
  }
  int actual_num = measured_traces.size();
  ThreadedTraceApply pp(self->postprocs_);
  std::vector<Schedule> results(actual_num, Schedule{nullptr});
  auto f_proc_measured = [this, &measured_traces, &results, &pp](int thread_id,
                                                                 int trace_id) -> void {
    PerThreadData& data = this->per_thread_data_.at(thread_id);
    TRandState* rand_state = &data.rand_state;
    const IRModule& mod = data.mod;
    tir::Trace trace = measured_traces.at(trace_id);
    Schedule& result = results.at(trace_id);
    ICHECK(!result.defined());
    if (Optional<Schedule> sch = pp.Apply(mod, trace, rand_state)) {
      result = sch.value();
    } else {
      LOG(FATAL) << "ValueError: Cannot postprocess the trace:\n" << trace;
      throw;
    }
  };
  support::parallel_for_dynamic(0, actual_num, self->ctx_->num_threads, f_proc_measured);
  return results;
}
        """
        _ = Profiler.timeit("EvoSearch/PickBestFromDatabase")
        measured_traces = []
        self.database_:Database
        top_records = self.database_.get_top_k(self.token_, num)
        for record in top_records:
            measured_traces.append(record.trace)
        actual_num = len(measured_traces)
        pp = ThreadedTraceApply(self.sraechstrategy.postprocs_)
        results = [None for _ in range(actual_num)]
        def f_proc_measured(thread_id, trace_id):
            data = self.sraechstrategy.per_thread_data_[thread_id]
            rand_state = data.rand_state
            mod = data.mod
            trace = measured_traces[trace_id]
            result = results[trace_id]
            assert result is None, f"result {trace_id} should be None"

            if sch := pp.apply(mod, trace, rand_state):
                result = sch
            else:
                LOG(FATAL) << "ValueError: Cannot postprocess the trace:\n" << trace
                raise


    def generatemeasurecandidates(self,):
        pass


@derived_object
class OurPySearchStrategy(PySearchStrategy):
    state: State = None
    context : TuneContext = None
    population_size = 512
    init_measured_ratio = 0.2
    init_min_unmeasured = 50
    max_fail_count = 5
    num_empty_iters_before_early_stop = 5
    genetic_num_iters = 4
    genetic_mutate_prob = 0.85
    genetic_max_fail_count = 10
    eps_greedy = 0.05


    def __init__(
        self,
        *,
        population_size = 512,
        init_measured_ratio = 0.2,
        init_min_unmeasured = 50,
        max_fail_count = 5,
        genetic_num_iters = 4,
        genetic_mutate_prob = 0.85,
        genetic_max_fail_count = 10,
        eps_greedy = 0.05)-> None:
        self.population_size = population_size
        self.init_measured_ratio = init_measured_ratio
        self.init_min_unmeasured = init_min_unmeasured
        self.max_fail_count = max_fail_count
        self.genetic_num_iters = genetic_num_iters
        self.genetic_mutate_prob = genetic_mutate_prob
        self.genetic_max_fail_count = genetic_max_fail_count
        self.eps_greedy = eps_greedy
        def check_probability(value):
            assert 0<=value<=1,"Probability must be in [0,1]"
        check_probability(self.init_measured_ratio)
        check_probability(self.genetic_mutate_prob)
        check_probability(self.eps_greedy)
    
    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the search strategy with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initialization.
        """
        initialize_with_tuneconext = tvm.get_global_func("meta_schedule.SearchStrategyInitializeWithTuneContext")
        return initialize_with_tuneconext(self,context)

    def pre_tuning(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"] = None,
        cost_model: Optional["CostModel"] = None,
    ) -> None:
        """Pre-tuning for the search strategy.

        Parameters
        ----------
        design_spaces : List[Schedule]
            The design spaces for pre-tuning.
        """
        pre_tuning = tvm.get_global_func("meta_schedule.SearchStrategyPreTuning")
        return pre_tuning(self,max_trials,num_trials_per_iter,design_spaces,database,cost_model)

    def post_tuning(self) -> None:
        """Post-tuning for the search strategy."""
        post_tuning = tvm.get_global_func("meta_schedule.SearchStrategyPostTuning")
        return post_tuning(self)

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generate measure candidates from design spaces for measurement.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if finished.
        """

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """Update the search strategy with profiling results.

        Parameters
        ----------
        measure_candidates : List[MeasureCandidate]
            The measure candidates for update.
        results : List[RunnerResult]
            The profiling results from the runner.
        """
        notify_runner_results = tvm.get_global_func("meta_schedule.SearchStrategyNotifyRunnerResults")
        return notify_runner_results(self,measure_candidates,results)

    def clone(self) -> SearchStrategy:
        """Clone the search strategy.

        Returns
        -------
        strategy : SearchStrategy
            The cloned search strategy.
        """
        clone = tvm.get_global_func("meta_schedule.SearchStrategyClone")
        return clone(self)