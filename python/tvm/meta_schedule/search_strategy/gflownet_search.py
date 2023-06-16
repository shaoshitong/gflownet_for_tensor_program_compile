
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Union

import tvm

# isort: off
from typing_extensions import Literal
# isort: on
import copy
import multiprocessing
from multiprocessing import Value
import tvm
import copy
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule, Trace

from .. import _ffi_api
from ..arg_info import ArgInfo
from ..cost_model import CostModel
from ..database import Database
from ..gflownet_utils.concurrentbitmask import ConcurrentBitmask
from ..gflownet_utils.heatmap import SizedHeap
from ..gflownet_utils.irmoduleset import IRModuleSet
from ..module_equality import E, ModuleEquality
from ..profiler import Profiler
from ..runner import RunnerResult
from ..tune_context import TuneContext
from ..utils import (cpu_count, derived_object,
                     get_global_func_with_default_on_worker)
from .search_strategy import MeasureCandidate, PySearchStrategy, SearchStrategy

"""
How to use it ?

sm = ModuleEquality("structural")
sm.equal(mod, mod)
sm.hash(mod)

"""

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

#zhangchunlei
from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.tir.schedule import Schedule

from ...tir.schedule import Trace
from .. import Profiler, _ffi_api, utils
from ..database.database import Workload
from ..postproc import Postproc
from ..space_generator.space_generator import SpaceGenerator
from .search_strategy import SearchStrategy, _PySearchStrategy

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..mutator import Mutator
    from ..tune_context import TuneContext

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

def list_swap(list1, list2):
    list1[:], list2[:] = list2[:], list1[:]


"""Need to implement"""
class PerThreadData:
    #auxiliary class for MyEvolutionarySearch
    mod :IRModule
    rand_state : np.int64
    trace_sampler : Callable[[], int]
    mutator_sampler : Callable[[Optional[Mutator]], None] 
    
    def __init__(self) -> None:
        self.mod = None
        self.rand_state = np.int64(-1)
        self.trace_sampler = None
        self.mutatot_sampler = None
    
    def Set(self, scores: List[float], genetic_mutate_prob:float, mutator_probs:Dict[Mutator, float]):
        from tvm import tir
        tir.multiply

    
    @staticmethod
    def default_trace_sampler(rand_state,weights,sum_type = "softmax"):
        np.random.seed(rand_state)
        if not isinstance(weights,np.ndarray):
            weights = np.array(weights)

        if sum_type == "linear":
            if weights.min() < 0:
                weights = weights - weights.min()
                weights = weights/np.sum(weights)
        elif sum_type == "softmax":
            weights = np.exp(weights - weights.min())/np.exp(weights - weights.min()).sum()
        else:
            raise NotImplementedError
        
        idx = np.random.choice(weights.shape[0],1,p=weights).item()
        return idx
    
    @staticmethod
    def default_mutatot_sampler(genetic_mutate_prob,mutator_probs,rand_state):
        np.random.seed(rand_state)
        mutators = []
        mutators.append(None)
        masses = []
        masses.append(1 - genetic_mutate_prob)
        total_mass_mutator = 0
        if genetic_mutate_prob>0:
            for mutator,mass in mutator_probs:
                total_mass_mutator += mass
                mutators.append(mutator)
                masses.append(mass*genetic_mutate_prob)
        if (total_mass_mutator == 0.0):
            masses[0] = 1.0
            for i in range(1,len(masses)):
                masses[i] = 0.0
        elif (total_mass_mutator != 1.0):
            for i in range(1,len(masses)):
                masses[i]/=total_mass_mutator
        
        return PerThreadData.default_trace_sampler(rand_state,masses)
        

        
class ThreadedTraceApply:


    class Item:
        postproc = None
        fail_counter = 0

        def __init__(self,postporc) -> None:
            self.postproc = postporc
    
    def __init__(self,postprocs) -> None:
        self.n_ = len(postprocs)
        self.items_ = [self.Item(postprocs[i]) for i in range(self.n_)]

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

    def SummarizeFailures(self):
        result = ""
        for i in range(self.n_):
            item = self.items_[i]
            result += "Postproc #"+str(i)+" ["+str(item.postproc)+"]: "+str(item.fail_counter.value)+" failure(s)"
            if i != self.n_ - 1:
                result += "\n"
        print(result)
        return result
    
def AssembleCandidates(picks):
  """Assemble a list of candidates from a list of schedules."""
  measure_inputs = [None for _ in range(len(picks))]
  for i, sch in enumerate(picks):
    measure_inputs[i] = MeasureCandidate(sch, ArgInfo.from_entry_func(sch.mod,remove_preproc=True))
  return measure_inputs

def PredictNormalizedScore(candidates,context,cost_model):
    """Predict the normalized score of a list of candidates."""
    _ = Profiler.timeit("EvoSearch/Evolve/PredictNormalizedScore")
    assert len(candidates) != 0, "Candidates given for score prediction can not be empty list!"
    scores = cost_model.predict(context, AssembleCandidates(candidates))
    scores = np.clip(scores,0.0,np.inf)
    return scores


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
    def __init__(self, context, searchstrategy, max_trials, num_trials_per_iter, design_space_schedules, database, cost_model, model_equality = "structural") -> None:
        self.context = context
        self.searchstrategy = searchstrategy
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_space_schedules = design_space_schedules
        self.database_ = database
        self.cost_model_ = cost_model
        self.model_equality = ModuleEquality(model_equality)
        self.st = 0
        self.ed = 0
        self.num_empty_iters = 0
        self.measured_workloads_:IRModuleSet = None
        self.design_spaces = []
        for space in self.design_space_schedules:
            self.design_spaces.append(space.trace.trace.simplified(True))
        self.mod = context.mod
        self.per_thread_data_ = [PerThreadData() for i in range(self.context.num_threads)]
        for i in range(self.context.num_threads):
            self.per_thread_data_[i].mod = copy.deepcopy(self.mod)
            self.per_thread_data_[i].rand_state = forkseed(self.sraechstrategy.rand_state_)
        self.token_ = database.commit_workload(self.mod)


    def reset(self):
        self.st = 0
        self.ed = 0
        self.num_empty_iters = 0
        self.measured_workloads_:IRModuleSet = None
        self.design_spaces = []
        for space in self.design_space_schedules:
            self.design_spaces.append(space.trace.trace.simplified(True))
        self.mod = self.context.mod
        self.per_thread_data_ = [PerThreadData() for i in range(self.context.num_threads)]
        for i in range(self.context.num_threads):
            self.per_thread_data_[i].mod = copy.deepcopy(self.mod)
            self.per_thread_data_[i].rand_state = forkseed(self.sraechstrategy.rand_state_)
        self.token_ = self.database_.commit_workload(self.mod)

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

    
    def ModuleHash(self, mod: IRModule)->int:
        return self.model_equality.hash(mod)
    
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
        
    def EvolveWithCostModel(self,population,num):
        self.database_:Database
        exists = IRModuleSet(self.model_equality)
        with Profiler.timeit("EvoSearch/Evolve/Misc/CopyMeasuredWorkloads"):
            assert num > 0, "num should be positive"
            exists = self.sraechstrategy.measured_workloads_
        iter = 0
        while True: 
            scores = PredictNormalizedScore(population,self.context,self.cost_model_)
            with Profiler.timeit("EvoSearch/Evolve/Misc"):
                assert len(scores) == len(population), "scores and population should have same length"
                # The heap to record best schedule, we do not consider schedules that are already measured
                heap = SizedHeap(num)
                for i in range(len(population)):
                    sch = population[i]
                    mod = sch.mod
                    shash = self.ModuleHash(mod)
                    score = scores[i]
                    if exists.Has(mod,shash) == False:
                        exists.Add(mod,shash)
                        heap.push((score,sch))
                if iter == self.searchstrategy.genetic_num_iters:
                    break
                for data in self.per_thread_data_:
                    data.Set(scores,self.searchstrategy.genertic_mutate_prob,self.searchstrategy.mutator_probs_)
            

            with Profiler.timeit("EvoSearch/Evolve/Mutation"):
                pp = ThreadedTraceApply(self.postprocs_)
                cbmask = ConcurrentBitmask(self.population_size)
                next_population = [None]*self.population_size

                def f_find_candidate(thread_id,trace_id):
                    data = self.per_thread_data_[thread_id]
                    rand_state = data.rand_state
                    mod = data.mod
                    trace_sampler = data.trace_sampler
                    mutator_sampler = data.mutator_sampler
                    result = next_population[trace_id]
                    sampled_trace_id = -1
                    for fail_count in range(self.searchstrategy.genetic_max_fail_count):
                        sampled_trace_id = trace_sampler()
                        trace = population[sampled_trace_id].trace
                        opt_mutator = mutator_sampler()
                        if opt_mutator:
                            mutator = opt_mutator.value
                            new_trace = mutator.Apply(trace,rand_state)
                            result = sch = pp.Apply(mod,new_trace,rand_state)
                            break
                    if result is None:
                        result = population[sampled_trace_id]
                    next_population[trace_id] = result

                pool = multiprocessing.Pool(processes=self.context.num_threads)
                pool.map(f_find_candidate, 0, self.searchstrategy.population_size)
                pool.close()
                pool.join()
                list_swap(population,next_population)

            iter+=1
        with Profiler.timeit("EvoSearch/Evolve/Misc"):
            # Return the best states from the heap, sorting from higher score to lower ones
            results = []
            for item in heap.heap[::-1]:
                results.append(item.sch)
        
        # output the tuning log
        kNumScoresPerLine = 16
        output_str = ""
        for st in range(0,len(heap.heap),kNumScoresPerLine):
            output_str += "\n"
            ed = min(st + kNumScoresPerLine,len(heap.heap))
            self.context.logger.info("[%d : %d]:\t",st + 1,ed)
            output_str += f"[{int(st+1)} : {int(ed)}]:\t"
            for i in range(st,ed):
                if i != st:
                    self.context.logger.info(" ")
                    output_str += " "
                self.context.logger.info("%g",heap.heap[i].score)
                output_str += f"{round(heap.heap[i].score,4)}"
            self.context.logger.info("\n")
            output_str += "\n"

        print(f"Scores of the best {len(heap.heap)} schedules:",output_str)
        return results
  

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
        context,
        population_size = 512,
        init_measured_ratio = 0.2,
        init_min_unmeasured = 50,
        max_fail_count = 5,
        genetic_num_iters = 4,
        genetic_mutate_prob = 0.85,
        genetic_max_fail_count = 10,
        eps_greedy = 0.05)-> None:

        assert context is not None,"context should not be None! It contains necessary information!"
        self.context = context
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

    
    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        assert context.num_threads > 0, "ValueError: `TuneContext.num_threads` must be > 0"
        assert self.context.space_generator is not None, "ValueError: `TuneContext.space_generator` must be defined"
        assert self.context.space_generator.postprocs is not None, "ValueError: `TuneContext.space_generator.postprocs` must be defined"
        assert self.context.space_generator.mutator_probs is not None, "ValueError: `TuneContext.space_generator.mutator_probs` must be defined"
        self.context = context
        self.postprocs = context.space_generator.postprocs
        self.mutator_probs = context.space_generator.mutator_probs
        self.rand_state = forkseed(context.rand_state)
        self.state = None

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
        assert design_spaces is not None, "Design space should not be None!"
        assert database is not None, "Context should not be None!"
        assert cost_model is not None, "Cost Model should not be None!"
        assert self.state is None, "ValueError: `PreTuning` is already invoked without corresponding `PostTuning`."
        self.state = State(self.context,self,max_trials,num_trials_per_iter,design_spaces,database,cost_model)

    def post_tuning(self) -> None:
        """Post-tuning for the search strategy."""
        self.state = None

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generate measure candidates from design spaces for measurement.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if finished.
        """
        return self.state.GenerateMeasureCandidates()


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
        self.state.NotifyRunnerResults(measure_candidates, results)

    def clone(self) -> SearchStrategy:
        """Clone the search strategy.

        Returns
        -------
        strategy : SearchStrategy
            The cloned search strategy.
        """
        copy_self = copy.deepcopy(self)
        copy_self.state = None
        return copy_self
