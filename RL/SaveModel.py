import torch

from MyPPO import PPO

def save(
    model:PPO,
    path,
    exclude = None,
    include = None,
) -> None:
    data = model.__dict__.copy()

    if exclude is None:
        exclude = []
    exclude = set(exclude).union(model._excluded_save_params())

    if include is not None:
        exclude = exclude.difference(include)

    state_dicts_names, torch_variable_names =model._get_torch_save_params()
    all_pytorch_variables = state_dicts_names + torch_variable_names
    for torch_var in all_pytorch_variables:
        var_name = torch_var.split(".")[0]
        exclude.add(var_name)

    for param_name in exclude:
        data.pop(param_name, None)

    params_to_save = model.get_parameters()
    torch.save(params_to_save, path)