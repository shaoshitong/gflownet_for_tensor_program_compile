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
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

task_scheduler = ms.task_scheduler.create("allpython")
cost_model = task_scheduler.get_costmodel()
database = ms.tune_tir(
    mod=MyModule,
    max_trials_global=64,
    num_trials_per_iter=64,
    target="llvm --num-cores=1",
    work_dir="./tune_tmp",
    task_name="main",
)
sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
a_nd = tvm.nd.array(np.random.uniform(size=(128, 128)).astype("float32"))
b_nd = tvm.nd.array(np.random.uniform(size=(128, 128)).astype("float32"))
c_nd = tvm.nd.empty((128, 128), "float32")
lib = tvm.build(sch.mod, target="llvm")
f_timer_after = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule after tuning: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
sch.trace.show()
IPython.display.HTML(code2html(sch.mod.script()))

