import json
import os
import numpy as np
import OpenEXR

# 读取lightmap文件样例
dataset_path = '../Data/SimpleData'
config_file = 'config.json'

# 读取数据集下的config文件，获取数据集配置参数
with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取lightmap数量
lightmap_count = data['lightmap_count']
print(f"Lightmap count: {lightmap_count}")
time = 1

# 获取每张lightmap的配置参数
for lightmap in data['lightmap_list']:
    id = lightmap['id']
    resolution = lightmap['resolution']
    parts = lightmap['parts']

    # 由于分辨率较大，一张lightmap可能由多个bin文件组成，读取该lightmap的所有bin文件后合并
    lightmap_bin_list = []
    mask_bin_list = []
    for part in range(parts):
        # 指定你想要读取的lightmap类型：DirectionalOnly、LocalOnly、SkyLightOnly、Original
        # lightmap_path = os.path.join(dataset_path, 'DirectionalOnly',f'path.bin')
        # lightmap_path = os.path.join(dataset_path, 'LocalOnly',f'path.bin')
        # lightmap_path = os.path.join(dataset_path, 'SkyLightOnly',f'path.bin')
        lightmap_path = os.path.join(dataset_path, 'Original',
                                     f'lightmapRawData_Dilate_VirtualTexture0_{id}_HQ_{resolution["width"]}_{resolution["height"]}_{time}.00.bin_{part}')
        mask_path = os.path.join(dataset_path, 'Mask',
                                 f'lightmapCoverage_Dilate_VirtualTexture0_{id}_HQ_{resolution["width"]}_{resolution["height"]}_1.00.bin_{part}')

        # 读取bin文件，lightmap数据类型为float32
        lightmap_part_bin = np.fromfile(lightmap_path, dtype=np.float32)
        lightmap_bin_list.append(lightmap_part_bin)
        # 当然如果你不需要整张的lightmap数据，只需要该part的数据
        # 直接将该部分数据reshape为对应形状即可，注意将高度除以parts：
        # lightmap_data = lightmap_part_bin.reshape(resolution['height'] // parts, resolution['width'], 3)
        
        # mask数据类型为int8
        mask_part_bin = np.fromfile(mask_path, dtype=np.int8)
        mask_bin_list.append(mask_part_bin)
    
    # 将不同part的bin文件合并，得到一张完整的lightmap以及mask数据
    lightmap_data = np.concatenate(lightmap_bin_list, axis=0)
    mask_data = np.concatenate(mask_bin_list, axis=0)

    # lightmap每个像素有R G B三通道
    lightmap = lightmap_data.reshape(resolution['height'], resolution['width'], 3)
    # mask每个像素有1通道
    mask = mask_data.reshape(resolution['height'], resolution['width'])

    # mask数据为-1时，表示该数据为无效数据，为127时，表示该数据为有效数据
    # 获取有效lightmap数据可以这样做：
    valid_lightmap = lightmap[mask >= 127]

    # 你可以将这些数据保存为exr文件以供查看：
    R = lightmap[:, :, 0].tobytes()
    G = lightmap[:, :, 1].tobytes()
    B = lightmap[:, :, 2].tobytes()
    exr_file = OpenEXR.OutputFile(f'lightmapRawData_Dilate_VirtualTexture0_{id}_HQ_{resolution["width"]}_{resolution["height"]}_{1}.00.exr', 
                                  OpenEXR.Header(resolution['width'], resolution['height']))
    exr_file.writePixels({'R': R, 'G': G, 'B': B})
    exr_file.close()

