import os
import glob
import math
import numpy as np
import yaml

def load_metrics(path):
    with open(path, "r") as yfile:
        string = yfile.read()
        return yaml.load(string, yaml.loader.BaseLoader)

def get_gtname(mname):
    return mname + "_gt"


def get_genname(mname):
    return mname + "_gen"


def get_reconsname(mname):
    return mname + "_recons"


def valformat(val, power=3):
    p = float(pow(10, power))
    # "{:<04}".format(np.round(p*val).astype(int)/p)
    return str(np.round(p*val).astype(int)/p).ljust(4, "0")


def format_values(values, key, latex=True):
    mean = np.mean(values)

    # if "accuracy" in key:
    #     mean = 100*mean
    #     values = 100*values
    #     smean = valformat(mean, 1)
    # else:
    # smean = valformat(mean, 3)

    # if "accuracy" in key:
    #     interval = valformat(1.96 * np.var(values), 4)  # [1:]
    # else:
    #     interval = valformat(1.96 * np.var(values), 4)  # [1:]

    # smean = valformat(mean, 2)

    if "accuracy" in key:
        interval = valformat(1.96 * np.var(values), 4)  # [1:]
        smean = valformat(mean, 3)
    else:
        interval = valformat(1.96 * np.var(values), 4)  # [1:]
        smean = valformat(mean, 3)
    
    if latex:
        string = rf"${smean}^{{\pm{interval}}}$"
    else:
        string = rf"{smean} +/- {interval}"
    return string


def print_results(folder, evaluation):
    evalpath = os.path.join(folder, evaluation)
    metrics = load_metrics(evalpath)

    a2m = metrics["feats"]

    has_split_keys = any(k.endswith("_train") or k.endswith("_test") for k in a2m)
    has_cd = any(k.startswith("cd_") for k in a2m)
    if "fid_gen_test" in a2m or has_split_keys:
        keys = [
            "fid_{}_train",
            "accuracy_{}_train",
            "multimodality_{}_train",
            "diversity_{}_train",
        ]
        if has_cd:
            keys.append("cd_{}_train")
        keys += [
            "fid_{}_test",
            "accuracy_{}_test",
            "multimodality_{}_test",
            "diversity_{}_test",
        ]
        if has_cd:
            keys.append("cd_{}_test")
    else:
        keys = ["fid_{}", "accuracy_{}", "diversity_{}", "multimodality_{}"]
        if has_cd:
            keys.append("cd_{}")

    model_names = set()
    for key in a2m:
        if key.startswith("fid_"):
            tail = key[len("fid_") :]
            if tail.endswith("_train"):
                model_names.add(tail[: -len("_train")])
            elif tail.endswith("_test"):
                model_names.add(tail[: -len("_test")])
            else:
                model_names.add(tail)

    if "fid_gt2" in a2m and "fid_gt" not in a2m:
        a2m["fid_gt"] = a2m["fid_gt2"]
        model_names.add("gt")

    order_priority = ["gt", "coarse", "refined", "gen", "recons"]
    lines = [name for name in order_priority if name in model_names]
    lines += sorted([name for name in model_names if name not in order_priority])

    rows = []
    rows_latex = []
    line_to_row_idx = {}

    for model in lines:
        row = ["{:6}".format(model)]
        row_latex = ["{:6}".format(model)]
        for key in keys:
            ckey = key.format(model)
            if ckey not in a2m:
                row.append("NA")
                row_latex.append("--")
                continue
            values = np.array([float(x) for x in a2m[ckey]])
            string_latex = format_values(values, key, latex=True)
            string = format_values(values, key, latex=False)
            row.append(string)
            row_latex.append(string_latex)
        rows.append(" | ".join(row))
        rows_latex.append(" & ".join(row_latex) + "\\")
        line_to_row_idx[model] = len(rows) - 1

    if "refined" in model_names and "coarse" in model_names:
        diff_row = ["{:6}".format("ref-c")]
        diff_row_latex = ["{:6}".format("ref-c")]
        has_all = True
        for key in keys:
            ckey_ref = key.format("refined")
            ckey_coarse = key.format("coarse")
            if ckey_ref not in a2m or ckey_coarse not in a2m:
                has_all = False
                break
            values_ref = np.array([float(x) for x in a2m[ckey_ref]])
            values_coarse = np.array([float(x) for x in a2m[ckey_coarse]])
            diff_values = values_ref - values_coarse
            diff_row.append(format_values(diff_values, key, latex=False))
            diff_row_latex.append(format_values(diff_values, key, latex=True))
        if has_all:
            insert_at = line_to_row_idx.get("refined", len(rows) - 1) + 1
            rows.insert(insert_at, " | ".join(diff_row))
            rows_latex.insert(insert_at, " & ".join(diff_row_latex) + "\\")

    table = "\n".join(rows)
    table_latex = "\n".join(rows_latex)
    print("Results")
    print(table)
    print()
    print("Latex table")
    print(table_latex)


if __name__ == "__main__":
    import argparse

    def parse_opts():
        parser = argparse.ArgumentParser()
        parser.add_argument("evalpath", help="name of the evaluation")
        return parser.parse_args()

    opt = parse_opts()
    evalpath = opt.evalpath

    folder, evaluation = os.path.split(evalpath)
    print_results(folder, evaluation)
