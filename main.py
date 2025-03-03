from run import main, get_args
import json
from pathlib import Path
from getpass import getuser

### AGENT RUN FUNCTIONS ###


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # SWE agent can only run on swebench
    assert 'model_name' in kwargs, 'model_name is required'
    
    # Set default values for kwargs
    per_instance_cost_limit = kwargs.get('per_instance_cost_limit', 3.0)
    skip_existing = kwargs.get('skip_existing', 'False')
    top_p = kwargs.get('top_p', '0.95')
    temperature = kwargs.get('temperature', '0.00')
    config_file = kwargs.get('config_file', Path(__file__).resolve().parent / "config" / "default.yaml")
    
    path = Path(__file__).resolve().parent / "input.json"
    # write the input to json file
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
    print(traj_dir)
    
    import os
    os.remove(path)
    
    with open(traj_dir) as f:
        data = list(f)
    model_name_or_path = traj_dir.split('/')[-2]
    print(model_name_or_path)

    for i, instance in enumerate(input):
        for line in data:
            line = json.loads(line)
            if instance['instance_id'] == line['instance_id']:
                instance['model_name_or_path'] = line['model_name_or_path']
                instance['model_patch'] = line['model_patch']
                break

            instance['model_name_or_path'] = model_name_or_path
            instance['model_patch'] = 'No patch returned'

    return input
    
    
# def run_swebench_gpt_4o_mini_c4(input):

#     # get path of this file
#     path = Path(__file__).resolve().parent / "input.json"
#     # write the input to json file
#     with open(path, "w") as f:
#         json.dump(input, f)

#     args = get_args(['--model_name', 'gpt-4o-mini-2024-07-18', '--data_path', str(path), '--per_instance_cost_limit', '1', '--skip_existing', 'False'])
#     main(args)

#     # delete the input file
#     import os
#     os.remove(path)

#     # # load all_preds.jsonl file in trajectories and add key data to input
#     with open(Path("trajectories") / Path(getuser()) / 'gpt-4o-mini-2024-07-18__input__default__t-0.00__p-0.95__c-1.00__install-1' / 'all_preds.jsonl', "r") as f:
#         data = list(f)

#     # for each instance_id in input dict, find the corresponding prediction in data
#     # and add it to the input dict
#     for i, instance in enumerate(input):
#         for line in data:
#             line = json.loads(line)
#             if instance['instance_id'] == line['instance_id']:
#                 instance['model_name_or_path'] = line['model_name_or_path']
#                 instance['model_patch'] = line['model_patch']
#                 break
#             instance['model_name_or_path'] = 'gpt-4o-mini-2024-07-18__input__default__t-0.00__p-0.95__c-1.00__install-1'
#             instance['model_patch'] = 'No patch returned'

#     return input
