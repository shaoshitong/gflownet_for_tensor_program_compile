
import tvm
from typing import TYPE_CHECKING, Callable, List, Optional, Union,Dict,Set

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
import multiprocessing

#zhangchunlei
import numpy as np
from tvm._ffi import register_object

from .. import _ffi_api
from .search_strategy import SearchStrategy
from .search_strategy import _PySearchStrategy
from ..space_generator.space_generator import SpaceGenerator
from ...tir.schedule import Trace
from tvm.ir import IRModule
from tvm.tir.schedule import Schedule
from ..database.database import Workload
from .. import Profiler
from .. import utils
from ..postproc import Postproc
import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..tune_context import TuneContext
    from ..mutator import Mutator
    


def forkseed(rand_state):
    rand_state = int(rand_state)
    return (rand_state * 32767) % 1999999973

def SampleInt(rand_state:np.int64, min_inclusive:int,max_exclusive:int):
    assert min_inclusive< max_exclusive, "ValueError: max_exclusive must be greater than min_inclusive."
    if(min_inclusive+1 == max_exclusive):
        return min_inclusive
    rand_ = forkseed(rand_state)
    dist = random.randint(min_inclusive, max_exclusive-1)
    return dist(rand_)

#rand)state是schedule的， 从0-n中采样k个（无重复）
def SampleWithoutReplacement(rand_state: np.int64, n:int, k:int)->List[int]:
    if k ==1:
        return SampleInt(rand_state, 0,n)
    if k == 2:
        result0 = SampleInt(rand_state,0,n)
        result1 = SampleInt(rand_state,0,n-1)
        if result1 >= result0:
            result1 += 1
        return [result0,result1]
    order = range(0,n)
    
    for i in range(k):
        j = SampleInt(rand_state,i ,n)
        if i != j:
            order[i], order[j] = order[j], order[i]
    return [order[0],order[k]]

def AssembleCandidates(picks:List[Schedule])->List[MeasureCandidate]:
    measure_inputs : List[MeasureCandidate]
    measure_inputs = [None]*len(picks)
    for sch in picks:
        measure_inputs.append(MeasureCandidate(sch,args_info=ArgInfo.from_entry_func(sch.mod(), remove_preproc= True)))
    return measure_inputs

# class Item:
#     def __init__(self,postproc) -> None:
#         self.postproc = postproc
#         self.fail_counter = Value('i', 0)
        
class ModuleEquality:
    def __init__(self) -> None:
        pass

class ModuleEquality:
    def __init__(self) -> None:
        pass
    def Equal(lhs:IRModule, rhs:IRModule):
        return True
#需要实现   
class IRModuleSet:
    
    class Item:
        mod_:IRModule
        shash_:int
        def __init__(self,mod:IRModule, shash:int) -> None:
            self.mod_ = mod
            self.shash_ = shash
        def __hash__(self) -> int:
            return self.shash_
        def __eq__(self, __value) -> bool:
            if isinstance(__value, IRModuleSet.Item):
                return self.shash_ == __value.shash_ and IRModuleSet.mod_eq_.Equal(self.mod_, __value.mod_)
            else:
                return False
            
    
    def __init__(self,mod_eq:ModuleEquality) -> None:
        self.tab_ : set[IRModuleSet.Item]= {}
        self.mod_eq_ :ModuleEquality = mod_eq
        
    def Add(self,mod: IRModule, shash:int):
        item = self.Item(mod,shash)
        self.tab_.add(item)
    def Has(self,mod:IRModule, shash:int):
        item = self.Item(mod,shash)
        return item in self.tab_
        

@register_object("meta_schedule.PerThreadData")  
class PerThreadData:
    #auxiliary class for MyEvolutionarySearch
    mod :IRModule
    rand_state : np.int64
    trace_sampler : Callable[[], int]
    mutator_sampler : Callable[[Optional[Mutator]], None] 
    
    def __init__(self) -> None:
        mod = None
        rand_state = np.int64(-1)
        trace_sampler = None
        mutatot_sampler = None
    
    def Set(scores: List[float], genetic_mutate_prob:float, mutator_probs:Dict[Mutator, float]):
        _ffi_api.EvolutionarySearchPerThreadDataSet(
        scores,
        genetic_mutate_prob,
        mutator_probs,
        )
        
    def MakeMutatorSampler(genetic_mutate_prob:float, mutator_probs:Dict[Mutator,float],rand_state:np.int64):
        _ffi_api.MakeMutatorSampler(
            genetic_mutate_prob,
            mutator_probs,
            rand_state,
        )        
        
        
class ThreadedTraceApply:
    def __init__(self,postprocs) -> None:
        self.n_ = len(postprocs)
        self.items_ = [Item(postprocs[i]) for i in range(self.n_)]

    def Apply(self,mod,trace,rand_state):
        sch = Schedule(mod,
                       forkseed(rand_state),
                       debug_mask=0,
                       error_render_level = "none")
        trace.apply_to_schedule(sch,remove_postproc=True)
        sch.enter_postproc()
        for i in range(self.n_):
            item = self.items_[i]
            if not item.postproc.apply(sch):
                item.fail_counter += 1
                return None
        return sch

    
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
        self.searchstrategy = searchstrategy
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_space_schedules = design_space_schedules
        self.database_ = database
        self.cost_model_ = cost_model
        self.st = 0
        self.ed = 0
        self.num_empty_iters = 0
        self.measured_workloads_:IRModuleSet = None
        self.design_spaces = []
        for space in self.design_space_schedules:
            self.design_spaces.append(space.trace.trace.simplified(True))
        self.mod = context.mod
        self.per_thread_data_ = []
        for i in range(self.context.num_threads):
            self.per_thread_data_[i].mod = copy.deepcopy(self.mod)
            self.per_thread_data_[i].rand_state = forkseed(self.sraechstrategy.rand_state_)
        self.token_ = database.commit_workload(self.mod)

    def pickbestfromdatabase(self,num) -> List[Schedule]:
        _ = Profiler.timeit("EvoSearch/PickBestFromDatabase")
        measured_traces = []
        self.database_:Database
        top_records = self.database_.get_top_k(self.token_, num)
        for record in top_records:
            measured_traces.append(record.trace)
        actual_num = len(measured_traces)
        pp = ThreadedTraceApply(self.searchstrategy.postprocs_)
        results = [None for _ in range(actual_num)]
        def f_proc_measured(thread_id, trace_id):
            data = self.searchstrategy.per_thread_data_[thread_id]
            rand_state = data.rand_state
            mod = data.mod
            trace = measured_traces[trace_id]
            result = results[trace_id]
            assert result is None, f"result {trace_id} should be None"
            sch = pp.apply(mod, trace, rand_state)
            if sch is not None:
                results[trace_id] = sch
            else:
                raise ValueError(f"Cannot postprocess the trace:\n{trace}")
        pool = multiprocessing.Pool(processes=self.context.num_threads)
        pool.map(f_proc_measured, range(actual_num))
        pool.close()
        pool.join()
        return results
    
    
    def SampleInitPopulation(self, num : int)-> List[Schedule]:
        _ = Profiler.timeit("EvoSearch/SampleInitPopulation")
        pp : ThreadedTraceApply(self.searchstrategy.postprocs)
        out_schs = []
        fail_count = 0
        while(len(out_schs) < self.searchstrategy.init_min_unmeasured and fail_count < self.sraechstrategy.max_fail_count):
            results = [None]*num
            def f_proc_unmeasured(thread_id:int, trace_id:int):
                data:PerThreadData                          
                data = self.per_thread_data_[thread_id]     
                rand_state = data.rand_state
                mod = data.mod
                assert  results[trace_id] is None , f"results {trace_id} should be None"
                design_space_index = SampleInt(rand_state,0,len(self.design_spaces))
                trace = Trace(self.design_spaces[design_space_index].insts, {})
                sch : Optional[Schedule] 
                sch = pp.Apply(mod, trace, rand_state)
                if sch is not None:
                    results[trace_id] = sch.get() #获取optional的值
            pool = multiprocessing.Pool(processes=self.context.num_threads)
            pool.map(f_proc_unmeasured, range(num))
            found_new  = False
            for i in range(num):
                if results[i] is not None:
                    found_new = True
                    out_schs.append(results[i])
            fail_count += not found_new
            self.context.logger.info('Sample-Init-Population summary:\n%s',pp.SummarizeFailures())
        return out_schs
    
    def EvolveWithCostModel(population,num):
        pass
    
    
    def ModuleHash(self,mod: IRModule)->int:
        #取决于 ModuleEquality的实现
        pass
    
    def PickWithEpsGreedy(self, unmeasured:List[Schedule], bests:List[Schedule], num:int)->List[Schedule]:
        """
        Pick final candidates from the given initial population and bests of evolved ones.

        Parameters:
        unmeasured: The initial population of traces sampled.
        bests: The best candidates predicted from evolved traces.
        num:   The number of traces to produce.

        Returns:    
        The final picked candidates with a ratio of both.
        """
        _ = Profiler.timeit("EvoSearch/PickWithEpsGreedy")
        num_rands = num * self.searchstrategy.eps_greedy#measurement采样的数量
        num_bests = num - num_rands
        rands =  SampleWithoutReplacement(self.context.rand_state, len(unmeasured), len(unmeasured))
        results:List[Schedule]
        measured_workloads = self.measured_workloads_
        i_bests, i_rands =0,0
        for i in range(num):
            has_best = i_bests < len(bests)
            has_rand = i_rands < len(rands)
            #pick schedule
            sch : Schedule = None
            if(i < num_bests):#need best
                if has_best:
                    i_bests+=1
                    sch = bests[i_bests]
                elif has_rand:
                    i_rands+=1
                    sch = unmeasured[rands[i_rands]]
                else:
                    break
            else: #not need best any more
                if has_rand:
                    i_rands+=1
                    sch = unmeasured[rands[i_rands]]
                elif(has_best):
                    i_bests+=1
                    sch = bests[i_bests]
                else:
                    break
            mod = sch.mod()
            shash = self.ModuleHash(mod)
            if(measured_workloads.Has(mod, shash) is False):
                measured_workloads.Add(mod, shash)
                results.append(sch)
        
        return results
            
    def GenerateMeasureCandidates(self)->Optional[List[MeasureCandidate]]:
        if(self.st >= self.max_trials):
            return None
        sample_num = self.num_trials_per_iter
        if self.ed > self.max_trials:
            sample_num = self.max_trials - self.st
            self.ed = self.max_trials
        assert self.st < self.ed, f"check fail: {self.st} < {self.ed}"
        pop = self.searchstrategy.population_size
        inits : List[Schedule]
        inits = [None]*pop
        
        self.context.logger.info("Generating candidates......")
        measured = self.pickbestfromdatabase(pop*self.searchstrategy.init_measured_ratio)
        self.context.logger.info("Picked top %s candidate(s) from database",len(measured))
        unmeasured :List[Schedule] = self.SampleInitPopulation(pop - len(measured))
        if(len(unmeasured) < self.searchstrategy.init_min_unmeasured):
            self.context.logger.warning("Cannot sample enough initial population, evolutionary search failed.")
            return None
        self.context.logger.info("Sample %s candidate(s)",len(unmeasured))
        inits.extend(measured)
        inits.extend(unmeasured)
        bests : List[Schedule] = self.EvolveWithCostModel(inits, sample_num)
        self.context.logger.info("Got %s candidate(s) with evolutionary search",len(bests))
        picks:List[Schedule] = self.PickWithEpsGreedy(unmeasured,bests,sample_num)
        self.context.logger.info("Sendding %s candidates(s) for measurement",len(picks))
        #判断是否为空，这里有一个空迭代容忍数量
        if(picks is None):
            self.num_empty_iters+=1
            if self.num_empty_iters >= self.searchstrategy.num_empty_iters_before_early_stop:
                return None
        return AssembleCandidates(picks)
    
    def NotifyRunnerResults(self, measure_candidates:List[MeasureCandidate],results:List[RunnerResult]):
        self.st += len(results)
        self.ed += len(results)
        
        

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
