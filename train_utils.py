import os
import torch


def load_checkpoint(checkpoint_path, model, criterion, optimizer):
    assert os.path.isfile(checkpoint_path)
    print(f'Loading checkpoint {checkpoint_path}')
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    if 'criterion' in checkpoint_dict:
        criterion.load_state_dict(checkpoint_dict['criterion'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print(f'Loaded checkpoint {checkpoint_path} from iteration {iteration}')
    return model, criterion, optimizer, learning_rate, iteration


def save_checkpoint(model, criterion, optimizer, learning_rate, iteration, filepath):
    print(f'Saving model and optimizer state at iteration {iteration} to {filepath}')
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def warm_start_model(checkpoint_path, model, ignore_layers):
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model

