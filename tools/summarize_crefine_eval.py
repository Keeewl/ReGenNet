import argparse
import csv
import json


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_results(payload):
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-json", required=True, type=str)
    parser.add_argument("--global-json", required=True, type=str)
    parser.add_argument("--csv-out", default="crefine_eval_summary.csv", type=str)
    parser.add_argument("--json-out", default="crefine_eval_summary.json", type=str)
    args = parser.parse_args()

    local_results = _extract_results(_load_json(args.local_json))
    global_results = _extract_results(_load_json(args.global_json))
    methods = sorted(set(local_results.keys()) | set(global_results.keys()))

    summary = {}
    for method in methods:
        summary[method] = {}
        summary[method].update(local_results.get(method, {}))
        summary[method].update(global_results.get(method, {}))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    keys = set()
    for metrics in summary.values():
        keys.update(metrics.keys())
    fieldnames = ["method"] + sorted(keys)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row = {"method": method}
            row.update(summary[method])
            writer.writerow(row)


if __name__ == "__main__":
    main()
