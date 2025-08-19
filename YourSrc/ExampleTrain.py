import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import os
import OpenEXR
import json

from ExampleModel import ExampleModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default='../Data/SimpleData')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
     
    config_file = 'config.json'
    time_count = 24
    with open(os.path.join(args.dataset, config_file), 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 分别训练每张lightmap
    for lightmap in config['lightmap_list']:
        print(f"training lightmap {lightmap['id']}")
        id = lightmap['id']
        resolution = lightmap['resolution']
        parts = lightmap['parts']

        # 读取每张lightmap在不同时间的数据
        # 关于读取数据，你应该参照ReadData.py来获取更详细的信息
        lightmap_in_different_time = []
        for time_idx in range(time_count):
            lightmap_part_bin_list = []
            for part in range(parts):
                lightmap_path = os.path.join(args.dataset, 'Original',
                                            f'lightmapRawData_Dilate_VirtualTexture0_{id}_HQ_{resolution["width"]}_{resolution["height"]}_{time_idx + 1}.00.bin_{part}')
                lightmap_part_bin = np.fromfile(lightmap_path, dtype=np.float32)
                lightmap_part_bin_list.append(lightmap_part_bin)
            lightmap_bin = np.concatenate(lightmap_part_bin_list, axis=0)
            lightmap_in_different_time.append(lightmap_bin.reshape(-1, 3))

        # 将数据拼接，转为torch.Tensor，传输到目标device上
        lightmap_data = torch.from_numpy(np.concatenate(lightmap_in_different_time, axis=0)).to(torch.float32).to(device)

        # 读取该张lightmap的mask数据
        mask_part_bin_list = []
        for part in range(parts):
            mask_path = os.path.join(args.dataset, 'Mask',
                                        f'lightmapCoverage_Dilate_VirtualTexture0_{id}_HQ_{resolution["width"]}_{resolution["height"]}_1.00.bin_{part}')
            mask_part_bin = np.fromfile(mask_path, dtype=np.int8)
            mask_part_bin_list.append(mask_part_bin)
        mask_data = np.concatenate(mask_part_bin_list, axis=0).reshape(resolution['height'], resolution['width'])

        # 初始化模型
        model = ExampleModel(input_dim=3, output_dim=3, hidden_dim=args.hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 生成归一化坐标
        xs, ys = np.meshgrid(np.arange(resolution['width']), np.arange(resolution['height']))
        coords = np.stack([ys / (resolution['height'] - 1), xs / (resolution['width'] - 1)], axis=-1).reshape(-1, 2)
        coords = torch.from_numpy(coords).to(torch.float32).to(device)
        total_coords = []
        for time_idx in range(time_count):
            alpha = torch.full((resolution['width'] * resolution['height'], 1), (time_idx / (time_count - 1))).to(device)
            coords_with_time = torch.cat([coords, alpha], dim=-1)
            total_coords.append(coords_with_time)
        total_coords = torch.cat(total_coords, dim=0)
        total_data = torch.cat([total_coords, lightmap_data], dim=-1)
        
        # 训练循环
        for it in range(args.iterations):
            batch_data = total_data[torch.randperm(total_data.shape[0])[:args.batch_size]]
            pred = model(batch_data[:, :3])
            loss = criterion(pred, batch_data[:, 3:])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (it + 1) % 10000 == 0:
                print(f"iteration {it + 1} loss: {loss.item()}")

        # 保存模型
        model.eval()
        os.makedirs(f"./ExampleResult", exist_ok=True)
        os.makedirs(f"./ExampleResult/images", exist_ok=True)

        torch.save(model.state_dict(), f"./ExampleResult/model_{id}.pth")
        with torch.no_grad():
            pred = model(total_coords[:, :3])
            
            pred = pred.reshape(time_count, resolution['height'], resolution['width'], 3).detach().cpu().numpy()
            lightmap_data = lightmap_data.reshape(time_count, resolution['height'], resolution['width'], 3).detach().cpu().numpy()

            # 计算PSNR
            psnr_list = []
            for time_idx in range(time_count):
                mse = np.mean((pred[time_idx][mask_data >= 127] - lightmap_data[time_idx][mask_data >= 127]) ** 2)
                psnr = 10 * np.log10(lightmap_data[time_idx][mask_data >= 127].max() ** 2 / mse)
                psnr_list.append(psnr)
            print(f"PSNR: {np.mean(psnr_list)}")
            
            # 保存拟合结果
            for time_idx in range(time_count):
                path = f'./ExampleResult/images/reconstructed_{id}_{time_idx + 1:02d}.00.exr'
                header = OpenEXR.Header(pred[time_idx].shape[1], pred[time_idx].shape[0])
                channels = ['R', 'G', 'B']
                exr = OpenEXR.OutputFile(path, header)
                exr.writePixels({
                    c: pred[time_idx][..., i].tobytes()
                    for i, c in enumerate(channels)
                })
                exr.close()
    
if __name__ == "__main__":
    main()