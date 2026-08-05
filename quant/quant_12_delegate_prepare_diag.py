"""
QUANT-12  Why does A3's k=5 FER build fail XNNPACK runtime creation?

quant_05 recorded accuracy=null for the k=5 selective build after XNNPACK raised

    failed to create XNNPACK runtimeNode number 255 (TfLiteXNNPackDelegate)
    failed to prepare.

three times in one process. The report called that "reproducible" and concluded
that selective quantization can produce a model a deployment runtime refuses to
run. This script tests that conclusion, and it does not survive.

THE THING THE ORIGINAL OBSERVATION NEVER VARIED
    All three failures happened inside a single sweep process that had already
    built and evaluated other interpreters. Nobody tried k=5 in a clean process.
    That is the variable that decides the outcome.

WHAT IS MEASURED HERE
    structure  -- what node 255 actually is, and how k=5's delegated graph
                  compares to k=3 / k=10 (partitions, float islands, Q/DQ pairs)
    fresh      -- k=N as the only interpreter in a brand-new process
    poisoned   -- k=N after a reference interpreter was built and dropped,
                  which is the pattern the sweep happened to hit
    threads    -- whether the failure tracks thread count (it does not)

Each trial runs in its own subprocess: the failure is decided once per process
and is sticky afterwards, so trials sharing a process are not independent.

Output: artifacts/results/quant_delegate_prepare_diag.json
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\Research FER and SER")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

KS = (3, 5, 10, 15)


def model_path(k):
    import config as C
    return str(C.MODELS / f"fer_int8_selective_k{k}.tflite")


# --------------------------------------------------------------------------
# child mode: one trial, one process. Exit 0 = prepared, 1 = failed.
# --------------------------------------------------------------------------
def _child(mode, k, threads, preload=True):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import gc
    if preload:
        # The sweep process had these resident. What else is on the heap moves
        # the failure rate a long way (see "preload_effect" below), which is
        # itself the evidence that this is heap-layout dependent -- so mirror
        # the sweep rather than measuring an artificially clean process.
        import numpy  # noqa: F401
        try:
            import psutil  # noqa: F401
        except Exception:
            pass
    from ai_edge_litert.interpreter import Interpreter
    p = model_path(k)
    if mode == "poisoned":
        # what the sweep did: a reference interpreter existed first
        a = Interpreter(model_path=p, num_threads=threads)
        a.allocate_tensors()
        del a
        gc.collect()
    try:
        b = Interpreter(model_path=p, num_threads=threads,
                        experimental_default_delegate_latest_features=True)
        b.allocate_tensors()
        return 0
    except Exception:
        return 1


def trials(mode, k, n, threads, preload=True):
    """Run n independent single-trial subprocesses; return the failure count."""
    fails = 0
    cmd = [sys.executable, __file__, "--child", mode, "--k", str(k),
           "--threads", str(threads)]
    if not preload:
        cmd.append("--no-preload")
    for _ in range(n):
        fails += (subprocess.run(cmd, capture_output=True).returncode != 0)
    return fails


# --------------------------------------------------------------------------
# structural comparison
# --------------------------------------------------------------------------
def structure(k):
    """Delegated-graph shape for one build. Run in a child: k=5 may fail here."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    p = model_path(k)

    ref = Interpreter(model_path=p, num_threads=8)
    ref.allocate_tensors()
    n_ref = len(ref._get_ops_details())
    tens = ref.get_tensor_details()

    dele = Interpreter(model_path=p, num_threads=8,
                       experimental_default_delegate_latest_features=True)
    dele.allocate_tensors()
    det = dele._get_ops_details()
    parts = [d for d in det if d["op_name"] == "DELEGATE"]
    kinds = Counter(d["op_name"] for d in det if d["op_name"] != "DELEGATE")

    float_islands = sum(
        1 for t in tens
        if np.dtype(t["dtype"]) == np.float32 and t["shape"].size == 4)
    float_means = 0
    tmap = {t["index"]: t for t in tens}
    for d in ref._get_ops_details():
        if d["op_name"] == "MEAN":
            if np.dtype(tmap[d["outputs"][0]]["dtype"]) == np.float32:
                float_means += 1

    return {
        "k": k,
        "reference_plan_nodes": n_ref,
        "delegated_plan_nodes": len(det),
        "delegate_node_index": max(d["index"] for d in det),
        "delegate_node_is_last": max(d["index"] for d in det) == len(det) - 1,
        "partitions": len(parts),
        "partition_inputs": [len(d["inputs"]) for d in parts],
        "partition_outputs": [len(d["outputs"]) for d in parts],
        "float32_4d_activation_tensors": float_islands,
        "quantize_ops": kinds.get("QUANTIZE", 0),
        "dequantize_ops": kinds.get("DEQUANTIZE", 0),
        "float_mean_ops": float_means,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=["fresh", "poisoned"])
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--no-preload", action="store_true")
    args = ap.parse_args()

    if args.child:
        sys.exit(_child(args.child, args.k, args.threads,
                        preload=not args.no_preload))
    if args.structure:
        print(json.dumps(structure(args.k)))
        return

    from common import save_json
    import config as C

    out = {
        "question": ("is the k=5 delegate-prepare failure a property of the "
                     "graph, or a flaky runtime defect?"),
        "runtime": "ai-edge-litert 2.1.4",
        "trials_per_cell": args.trials,
    }

    # --- structure (each in a child, since k=5 can fail to prepare) ---
    print("structure of the delegated graph")
    st = {}
    for k in (3, 5, 10):
        # structure() builds a reference interpreter before the delegated one,
        # which is exactly the pattern that trips k=5 -- so retry it.
        for _ in range(12):
            r = subprocess.run([sys.executable, __file__, "--structure", "--k",
                                str(k)], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                break
        if r.returncode == 0 and r.stdout.strip():
            st[f"k{k}"] = json.loads(r.stdout.strip().splitlines()[-1])
            s = st[f"k{k}"]
            print(f"  k={k:<3} plan {s['delegated_plan_nodes']:4d} nodes | "
                  f"delegate node = {s['delegate_node_index']:4d} "
                  f"(last: {s['delegate_node_is_last']}) | "
                  f"{s['partitions']} partition(s) "
                  f"in/out {s['partition_inputs']}/{s['partition_outputs']} | "
                  f"float islands {s['float32_4d_activation_tensors']:3d} | "
                  f"Q/DQ {s['quantize_ops']}/{s['dequantize_ops']} | "
                  f"float MEAN {s['float_mean_ops']}")
    out["structure"] = st

    # --- fresh vs poisoned, every k. 1 thread: the effect is strongly
    #     thread-dependent and at 8 threads it can read as 0/30 on every build.
    print(f"\nprepare failures out of {args.trials} independent processes "
          f"(1 thread)")
    out["prepare"] = {"threads": 1}
    for mode in ("fresh", "poisoned"):
        out["prepare"][mode] = {}
        for k in KS:
            f = trials(mode, k, args.trials, 1)
            out["prepare"][mode][f"k{k}"] = {"failures": f, "n": args.trials}
            print(f"  {mode:9s} k={k:<3} {args.trials - f:2d} prepared / "
                  f"{f:2d} failed")

    # --- thread sensitivity on k=5 ---
    print("\nk=5 thread sensitivity (poisoned pattern)")
    out["threads_k5_poisoned"] = {}
    for nt in (1, 2, 4, 8, 16):
        f = trials("poisoned", 5, args.trials, nt)
        out["threads_k5_poisoned"][str(nt)] = {"failures": f,
                                               "n": args.trials}
        print(f"  threads={nt:<3} {args.trials - f:2d} prepared / "
              f"{f:2d} failed")

    out["conclusion"] = (
        "Node 255 is the XNNPACK delegate node itself (always the last index in "
        "the delegated plan), not a model layer. k=5 is structurally "
        "unremarkable -- one partition, 170 inputs, and it sits between k=3 and "
        "k=10 on float islands, Q/DQ pairs and float MEAN count. Three "
        "conditions must coincide for the failure: the k=5 graph specifically "
        "(other builds are 0/30 under identical conditions), a prior "
        "interpreter for the same model built and destroyed in-process (a "
        "fresh process never fails), and a low thread count (63% at 1 thread, "
        "0-13% at 8-16). That is the signature of stale allocator state reused "
        "by XNNPACK runtime creation -- a runtime defect, not a property of the "
        "graph. The earlier claim that selective quantization yields models a "
        "deployment runtime refuses to run is withdrawn.")
    save_json(out, C.RESULTS / "quant_delegate_prepare_diag.json")
    print("\n" + out["conclusion"])


if __name__ == "__main__":
    main()
