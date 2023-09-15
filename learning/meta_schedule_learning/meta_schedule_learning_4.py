import IPython
import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm.ir.module import IRModule
from tvm.script import tir as T


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
                    C[vi, vj] == 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


task_scheduler = ms.task_scheduler.create("allpython")
cost_model = task_scheduler.get_costmodel()
database = ms.tune_tir(
    mod=MyModule,
    max_trials_global=64,
    num_trials_per_iter=64,
    strategy = "evolutionary", #"evolutionary"
    target="cuda --max_threads_per_block 1024 --thread_warp_size 32 --max_shared_memory_per_block 49152",
    work_dir="./tune_tmp",
    task_name="main",
)
sch = ms.tir_integration.compile_tir(database, MyModule, "cuda --max_threads_per_block 1024 --thread_warp_size 32 --max_shared_memory_per_block 49152")
a_nd = tvm.nd.array(np.random.uniform(size=(128, 128)).astype("float32"))
b_nd = tvm.nd.array(np.random.uniform(size=(128, 128)).astype("float32"))
c_nd = tvm.nd.empty((128, 128), "float32")
lib = tvm.build(sch.mod, target="cuda --max_threads_per_block 1024 --thread_warp_size 32 --max_shared_memory_per_block 49152")
IPython.display.HTML(code2html(sch.mod.script()))

