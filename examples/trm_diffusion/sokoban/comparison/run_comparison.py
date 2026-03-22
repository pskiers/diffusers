import os
import yaml
import subprocess
import copy


def dict_to_hydra_args(d, prefix=""):
    args = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            args.extend(dict_to_hydra_args(v, key + "."))
        elif isinstance(v, list):
            formatted_list = str(v).replace(' ', '')
            args.append(f"{key}={formatted_list}")
        elif v is None:
            args.append(f"{key}=null")
        else:
            args.append(f"{key}={v}")
    return args


def submit_slurm_job(model_name, task_name, base_config, task_config, model_config):
    run_name = f"{model_name}-{task_name}"

    experiment_config = copy.deepcopy(base_config)
    for key, value in copy.deepcopy(task_config).items():
        experiment_config[key] = value
    if "model" not in experiment_config:
        experiment_config["model"] = {}
    for key, value in copy.deepcopy(model_config).items():
        experiment_config["model"][key] = value
    experiment_config["output_dir"] = run_name

    hydra_args = dict_to_hydra_args(experiment_config)

    sbatch_cmd = [
        "sbatch",
        f"--job-name={run_name}",
        "run_single.slurm"
    ] + hydra_args

    print(f"Sent task to slurm: {run_name}")

    process = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(f"Success in creating slurm task: {stdout.strip()}")
    else:
        print(f"Error in creating slurm task {run_name}:\n{stderr}")


def main():
    config_path = "comparison-config.yaml"
    with open(config_path, "r") as f:
        main_config = yaml.safe_load(f)

    tasks = {name: cfg["properties"] for name, cfg in main_config["tasks"].items() if cfg.get("active", False)}
    trm_models = {f"trm-{name}": cfg["properties"] for name, cfg in main_config["models"]["trm"].items() if cfg.get("active", False)}
    standard_models = {f"standard-{name}": cfg["properties"] for name, cfg in main_config["models"]["standard"].items() if cfg.get("active", False)}
    models = {**trm_models, **standard_models}

    base_config = copy.deepcopy(main_config)
    base_config.pop("tasks", None)
    base_config.pop("models", None)

    for task_name, task_config in tasks.items():
        for model_name, model_config in models.items():
            if task_name == 'cond-single-k':
                for k in task_config["k"]:
                    updated_task_config = copy.deepcopy(task_config)
                    updated_task_config["k"] = k
                    submit_slurm_job(model_name, f"{task_name}-k{k}", base_config, updated_task_config, model_config)
            else:
                submit_slurm_job(model_name, task_name, base_config, task_config, model_config)

    print("\nAll experiments were sent to slurm.")


if __name__ == "__main__":
    main()
