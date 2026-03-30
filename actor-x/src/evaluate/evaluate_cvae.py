from src.parser.evaluation import parser
import os


def _load_interx_action_names(data_path):
    candidates = []
    if data_path:
        abs_path = os.path.abspath(data_path)
        dataset_dir = os.path.dirname(os.path.dirname(abs_path))
        candidates.append(os.path.join(dataset_dir, "annots", "action_setting.txt"))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates.append(os.path.join(repo_root, "dataset", "interx", "annots", "action_setting.txt"))

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    return []


def main():
    parameters, folder, checkpointname, epoch, niter = parser()

    dataset = parameters["dataset"]
    print(dataset)
    model_path = parameters["model_path"]
    if dataset == "chi3d":
        from .stgcn_eval import evaluate
        num_classes = 8
        num_person = 2
    elif dataset == "interx":
        from .stgcn_eval import evaluate
        action_names = _load_interx_action_names(parameters.get("datapath"))
        if not action_names:
            raise ValueError("InterX action_setting.txt not found or empty.")
        num_classes = len(action_names)
        num_person = 2
    else:
        raise NotImplementedError("This dataset is not supported.")

    evaluate(parameters, folder, checkpointname, epoch, niter, num_classes, model_path, num_person)


if __name__ == "__main__":
    main()
