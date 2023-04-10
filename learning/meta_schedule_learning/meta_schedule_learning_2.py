from tvm import meta_schedule as ms

import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

import IPython


def code2html(code):
    """Helper function to use pygments to turn the code string into highlighted html."""
    import pygments
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import Python3Lexer
    formatter = HtmlFormatter()
    html = pygments.highlight(code, Python3Lexer(), formatter)
    return "<style>%s</style>%s\n" % (formatter.get_style_defs(".highlight"), html)

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


mod=MyModule
max_trials_global=64
num_trials_per_iter=64
target="llvm --num-cores=1"
work_dir="./tune_tmp"
task_name="main"
seed = None
space = "post-order-apply"
strategy = "evolutionary"
num_tuning_cores = "physical"
builder = "local"
runner = "local"
database = "json"
cost_model = "xgb"
measure_callbacks = "default"
task_scheduler = "round-robin"
module_equality = "structural"
from tvm.meta_schedule.logging import get_loggers_from_work_dir
from tvm.meta_schedule.utils import fork_seed
from tvm.meta_schedule.tune_context import TuneContext, _normalize_mod
from tvm.meta_schedule.builder import Builder
from tvm.meta_schedule.cost_model import CostModel
from tvm.meta_schedule.database import Database
from tvm.meta_schedule.measure_callback import MeasureCallback
from tvm.meta_schedule.runner import Runner
from tvm.meta_schedule.task_scheduler import TaskScheduler,PyTaskScheduler
from tvm.meta_schedule.tune_context import TuneContext


(logger,) = get_loggers_from_work_dir(work_dir, [task_name])
(seed,) = fork_seed(seed, n=1)
tasks = [
        TuneContext(
            mod=mod,
            target=target,
            space_generator=space,
            search_strategy=strategy,
            task_name=task_name,
            logger=logger,
            rand_state=seed,
            num_threads=num_tuning_cores,
        ).clone()]

task_weights=[1.0]
work_dir=work_dir
max_trials_global=max_trials_global
max_trials_per_task=max_trials_global
num_trials_per_iter=num_trials_per_iter
builder=builder
runner=runner
database=database
cost_model=cost_model
measure_callbacks=measure_callbacks
task_scheduler=task_scheduler


if len(tasks) == 0:
    raise ValueError("No tasks to tune.")

if len(tasks) != len(task_weights):
    raise ValueError(
        f"Length of tasks ({len(tasks)}) and task_weights ({len(task_weights)}) do not match."
    )

num_cores = tasks[0].num_threads

if max_trials_per_task is None:
    max_trials_per_task = max_trials_global
if not isinstance(builder, Builder):
    builder = Builder.create(builder, max_workers=num_cores)
if not isinstance(runner, Runner):
    runner = Runner.create(runner, max_workers=num_cores)
if database == "json":
    database = Database.create(database, work_dir=work_dir, module_equality=module_equality)
elif not isinstance(database, Database):
    database = Database.create(database, module_equality=module_equality)
if not isinstance(cost_model, CostModel):
    cost_model = CostModel.create(cost_model, num_tuning_cores=num_cores)
if isinstance(measure_callbacks, MeasureCallback):
    measure_callbacks = [measure_callbacks]
elif measure_callbacks == "default":
    measure_callbacks = MeasureCallback.create(measure_callbacks)
# if not isinstance(task_scheduler, TaskScheduler):
#     task_scheduler = TaskScheduler.create(task_scheduler)
from tvm.meta_schedule.task_scheduler.task_scheduler import create
task_scheduler = create("allpython")
designspace  = task_scheduler.tune_designspace(
    tasks=tasks,
    task_weights=task_weights,
    max_trials_global=max_trials_global,
    max_trials_per_task=max_trials_per_task,
    num_trials_per_iter=num_trials_per_iter,
    builder=builder,
    runner=runner,
    measure_callbacks=measure_callbacks,
    database=database,
    cost_model=cost_model,
)
from tvm.tir import Schedule
for task_designspace in designspace:
    for sub_designspace in task_designspace:
        sub_schedule: Schedule = sub_designspace
        print(sub_schedule.trace)
        print("="*120)
# task_scheduler.tune(
#     tasks=tasks,
#     task_weights=task_weights,
#     max_trials_global=max_trials_global,
#     max_trials_per_task=max_trials_per_task,
#     num_trials_per_iter=num_trials_per_iter,
#     builder=builder,
#     runner=runner,
#     measure_callbacks=measure_callbacks,
#     database=database,
#     cost_model=cost_model,
# )

# sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
# a_nd = tvm.nd.array(np.random.uniform(size=(128, 128)).astype("float32"))
# b_nd = tvm.nd.array(np.random.uniform(size=(128, 128)).astype("float32"))
# c_nd = tvm.nd.empty((128, 128), "float32")
# lib = tvm.build(sch.mod, target="llvm")
# f_timer_after = lib.time_evaluator("main", tvm.cpu())
# print("Time cost of MyModule after tuning: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
# sch.trace.show()
# IPython.display.HTML(code2html(sch.mod.script()))

