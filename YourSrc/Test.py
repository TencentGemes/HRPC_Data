import json
import os
import time
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage
import lpips

import Interface

def cal_psnr(lightmap, lightmap_reconstruct, mask):
    mse = np.mean((lightmap[mask >= 127] - lightmap_reconstruct[mask >= 127]) ** 2)
    max_value = np.max(lightmap[mask >= 127])
    psnr = 10 * np.log10(max_value ** 2 / mse)
    return psnr

def cal_ssim(lightmap, lightmap_reconstruct, mask):
    h, w, _ = lightmap.shape
    lightmap_reconstruct[mask <= 0] = 0
    ssim_values = []
    for channel in range(3):
        img1 = lightmap[..., channel].reshape(h, w)
        img2 = lightmap_reconstruct[..., channel].reshape(h, w)
        ssim_val = ssim_skimage(img1, img2, data_range=img1.max() - img1.min())
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

loss_fn = lpips.LPIPS(net='alex')
def cal_lpips(lightmap, lightmap_reconstruct, mask):    
        h, w, _ = lightmap.shape
        lightmap_reconstruct[mask <= 0] = 0
        lightmap = (lightmap - lightmap.min()) / (lightmap.max() - lightmap.min())
        lightmap_reconstruct = (lightmap_reconstruct - lightmap_reconstruct.min()) / (lightmap_reconstruct.max() - lightmap_reconstruct.min())
        
        lightmap_tensor = torch.from_numpy(lightmap).permute(2, 0, 1).unsqueeze(0).float()
        lightmap_reconstruct_tensor = torch.from_numpy(lightmap_reconstruct).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            lpips_value = loss_fn(lightmap_tensor, lightmap_reconstruct_tensor).item()
        
        return lpips_value

if __name__ == '__main__': 
    # 所有大赛使用的数据集
    data_set_list = ['SimpleData']

    for data_set in data_set_list:
        dataset_path = f'../Data/{data_set}'
        config_file = 'config.json'
        with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        lightmap_count = data['lightmap_count']

        psnr_list = []
        ssim_list = []
        lpips_list = []
        time_list = []

        # 计算所有lightmap的指标
        for lightmap in data['lightmap_list']:
            lightmap_id = lightmap['id']
            resolution = lightmap['resolution']
            parts = lightmap['parts']
            
            # 我们可能会这样调用你的接口：
            xs, ys = np.meshgrid(np.arange(resolution['width']), np.arange(resolution['height']))
            coords = np.stack([ys, xs], axis=-1).reshape(-1, 2)

            for current_time in range(24):
                coords_with_time = np.concatenate([coords, np.full((coords.shape[0], 1), current_time+1)], axis=-1)

                # 调用参赛者提供的接口
                time_start = time.time()
                lightmap_reconstruct = Interface.reconstruct(data_set, lightmap_id, coords_with_time)
                time_end = time.time()
                time_list.append(time_end - time_start)
                lightmap_reconstruct = lightmap_reconstruct.reshape(resolution['height'], resolution['width'], 3)

                # 读取原始lightmap数据计算指标
                lightmap_bin_list = []
                mask_bin_list = []
                for part in range(parts):
                    lightmap_path = os.path.join(dataset_path, 'Original',
                                                f'lightmapRawData_Dilate_VirtualTexture0_{lightmap_id}_HQ_{resolution["width"]}_{resolution["height"]}_{current_time+1}.00.bin_{part}')
                    mask_path = os.path.join(dataset_path, 'Mask',
                                            f'lightmapCoverage_Dilate_VirtualTexture0_{lightmap_id}_HQ_{resolution["width"]}_{resolution["height"]}_1.00.bin_{part}')

                    lightmap_part_bin = np.fromfile(lightmap_path, dtype=np.float32)
                    lightmap_bin_list.append(lightmap_part_bin)
                    mask_part_bin = np.fromfile(mask_path, dtype=np.int8)
                    mask_bin_list.append(mask_part_bin)
                
                lightmap_data = np.concatenate(lightmap_bin_list, axis=0)
                mask_data = np.concatenate(mask_bin_list, axis=0)
                lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
                mask = mask_data.reshape(resolution['height'], resolution['width'])

                # 每128*128分辨率计算一次指标，最后取平均值
                # 我们在计算指标的时候不会考虑无效的像素，所以你可以在你的训练中不考虑这些无效像素
                part_size = 128
                for i in range(lightmap.shape[0] // part_size):
                    for j in range(lightmap.shape[1] // part_size):
                        lightmap_part = lightmap[i * part_size:(i + 1) * part_size, j * part_size:(j + 1) * part_size, :]
                        lightmap_reconstruct_part = lightmap_reconstruct[i * part_size:(i + 1) * part_size, j * part_size:(j + 1) * part_size, :]
                        mask_part = mask[i * part_size:(i + 1) * part_size, j * part_size:(j + 1) * part_size]
                        valid_mask = mask_part >= 127
                        if (np.any(valid_mask) and lightmap_part.max() != 0):
                            psnr_list.append(cal_psnr(lightmap_part, lightmap_reconstruct_part, mask_part))
                            ssim_list.append(cal_ssim(lightmap_part, lightmap_reconstruct_part, mask_part))
                            lpips_list.append(cal_lpips(lightmap_part, lightmap_reconstruct_part, mask_part))

        print(f"Data Set: {data_set}")
        print(f"PSNR: {np.mean(psnr_list)}")
        print(f"SSIM: {np.mean(ssim_list)}")
        print(f"LPIPS: {np.mean(lpips_list)}")
        print(f"Time: {np.sum(time_list)}")


