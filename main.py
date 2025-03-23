from run import main, get_args
import json
from pathlib import Path
from getpass import getuser


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # SWE agent can only run on swebench
    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'
    
    # Set default values for kwargs
    per_instance_cost_limit = kwargs.get('per_instance_cost_limit', 3.0)
    skip_existing = kwargs.get('skip_existing', 'False')
    top_p = kwargs.get('top_p', '0.95')
    temperature = kwargs.get('temperature', '0.00')
    config_file = kwargs.get('config_file', Path(__file__).resolve().parent / "config" / "default.yaml")
    path = Path(__file__).resolve().parent / "input.json"
    
    
    # Reformat the input.json file to make it compatible with the SWE-agent
    tasks = []
    for instance_id, task in input.items():
        tasks.append(task)
    with open(path, "w") as f:
        json.dump(tasks, f, indent=4)

    args = get_args([
        '--model_name', kwargs['model_name'],
        '--data_path', str(path),
        '--per_instance_cost_limit', f"{float(per_instance_cost_limit):.2f}",
        '--skip_existing', skip_existing,
        '--top_p', top_p,
        '--temperature', temperature,
        '--config_file', str(config_file)
    ])
    
    print(args)
    
    traj_dir = main(args) / "all_preds.jsonl"
    
    
    with open(traj_dir) as f:
        data = list(f)
    keys = list(input.keys())
    print(keys)
    
    model_patch = None
    for idx, key in enumerate(keys):
        for line in data:
            line = json.loads(line)
            instance = input[key]
            print(instance)
            print(line)

            if instance['instance_id'] == line['instance_id']:
                print("find matching instance!")
                model_patch = line['model_patch']
                break
    
    print(model_patch)
    return {input[keys[0]]['instance_id']: model_patch}