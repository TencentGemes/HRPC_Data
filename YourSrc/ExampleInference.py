import torch
import json
import os

from ExampleModel import ExampleModel

def read_model(model_path, hidden_dim, device):
    model = ExampleModel(input_dim=3, output_dim=3, hidden_dim=hidden_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def inference(dataset, lightmap_id, coords):
    if dataset != "SimpleData":
        return None
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = read_model(f"./ExampleResult/model_{lightmap_id}.pth", 256, device)

        dataset_path = f'../Data/{dataset}'
        config_file = 'config.json'
        with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        lightmap_list = data['lightmap_list']
        for lightmap in lightmap_list:
            if lightmap['id'] == lightmap_id:
                resolution = lightmap['resolution']
                break
        height = resolution['height']
        width = resolution['width']
        time_count = 24

        coords[:, 0] = coords[:, 0] / (height - 1)
        coords[:, 1] = coords[:, 1] / (width - 1)
        coords[:, 2] = (coords[:, 2] - 1) / (time_count - 1)
        coords = torch.from_numpy(coords).to(torch.float32).to(device)
        pred = model(coords)
        return pred.detach().cpu().numpy()

