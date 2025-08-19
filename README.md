# 腾讯游戏首届高性能渲染大赛
本次大赛主要由腾讯魔方工作室群《暗区突围》团队和腾讯游戏前沿技术MagicDawn团队承办。
## 数据说明

#### 文件类型
我们提供两种文件，包含光照数据文件以及掩码文件 <br>
掩码文件标识了对应区域的光照数据是否有效，无效位置光照忽略即可
1. **光照数据文件** (`lightmapRawData_*`)
    - 包含RGB三通道的光照信息
    - 数据类型：`float32`
    - 每个像素3个float32值（R、G、B）
    - 数据按照行优先存储，RGB按顺序存储
    - 文件大小: `width × height × 3 × 4 bytes`
    - 包含1-24点，整点时刻的lightmap数据

2. **掩码文件** (`lightmapCoverage_*`)
   - 标识有效/无效像素区域
   - 数据类型：`int8`
   - 每个像素1个int8值
   - 数据按照行优先存储
   - 文件大小: `width × height × 1 byte`

#### 光照类型
为了更灵活的算法处理，我们同样提供分离的光照数据，请按照你的算法设计使用所需要的光照数据 <br>
数据集下不同的文件夹内存放不同类型的光照数据 <br>
Original = DirectionalOnly + LocalOnly + SkyLightOnly
- **Original**: 原始完整光照数据
- **DirectionalOnly**: 仅方向光数据
- **LocalOnly**: 仅局部光照数据
- **SkyLightOnly**: 仅天空光数据

#### 掩码值含义
- **-1**: 无效数据像素
- **127**: 有效数据像素

#### 读取示例
请参照脚本ReadData.py

#### 数据集说明
数据集结构示意如下：<br>
config文件中含有该数据集的相关信息： <br>

- **lightmap_count** 表示该数据集中共有多少张lightmap <br>
- **lightmap_list** 对应每张lightmap信息，其中包括：<br>
    - **id**：lightmap对应的id <br>
    - **resolution**：lightmap对应的分辨率 <br>
    - **parts**：该lightmap文件被分为多少个文件保存 <br>

📁OldFactory <br>
├──  config.json <br>
├──  📁Mask <br>
│ ├── lightmapCoverage_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_0 <br>
│ ├── lightmapCoverage_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_1 <br>
│ ├── lightmapCoverage_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_2 <br>
│ ├── lightmapCoverage_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_3 <br>
│ └── lightmapCoverage_Dilate_VirtualTexture0_2_HQ_4096_4096_1.00.bin_0 <br>
├──  📁Original <br>
│ ├── lightmapRawData_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_0 <br>
│ ├── lightmapRawData_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_1 <br>
│ ├── lightmapRawData_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_2 <br>
│ ├── lightmapRawData_Dilate_VirtualTexture0_1_HQ_16384_16384_1.00.bin_3 <br>
│ ├── lightmapRawData_Dilate_VirtualTexture0_2_HQ_4096_4096_1.00.bin_0 <br>
├──  📁DirectionalOnly <br>
│ └── [类似文件结构] <br>
├──  📁LocalOnly <br>
│ └── [类似文件结构] <br>
├──  📁SkyLightOnly <br>
│ └── [类似文件结构] <br>


### 接口说明
你需要完善Interface.py中的reconstruct函数，以实现你的算法的解压部分 <br>
请不要在Interface中引入太多额外python第三方库，如果特殊需要请说明 <br>
尽量在该函数中进行exe文件的调用，以减少第三方环境的影响 <br>
注意，你需要提供Random-Access的实现方式，我们可能会随机生成采样坐标进行测试 <br>
（Random-Access：指的是可以直接跳转到任意位置访问数据，访问时间基本与位置无关，不需要先经过访问其他的数据） <br>
你可以参照Test.py脚本来简单测试你的接口是否可用，建议你修改Test.py脚本来保存你解压生成的lightmap，观察结果是否对齐并符合你的预期 <br>

在该函数中，你需要根据输入的dataset, lightmap_id以及coords，返回对应点的lightmap的RGB值 <br>
dataset为数据集名称，请处理所有大赛给出的数据集: ['OldFactory', ...] <br>
lightmap_id与数据集中lightmap_list中的id对应，为整数 <br>
coords为一个numpy数组，形状为[N, 3]，表示输入N个坐标点，每个坐标点有y, x, time三个分量 <br>
y对应lightmap纵坐标，x代表横坐标，并且y, x为非归一化的坐标，代表像素的整数坐标，如[0, 0]代表最左上角像素，[0, 50]代表第一行的第51个像素 <br>
time代表时间，time为 0 <= time <= 24 的浮点数 <br>

我们提供了一个简单的数据集以及训练脚本，你可以参考训练以及推理过程来理解我们的任务 <br>
ExampleModel.py定义了一个简单的MLP模型，输入是lightmap的坐标以及当前时间（三维），输出lightmap对应位置的RGB值（三维） <br>
ExampleTrain.py展示了一个简单的训练流程，从读取数据到优化再到保存结果，训练策略不具有参考价值，请制定你自己的训练策略，例如一个MLP模型具体对应多大的lightmap范围 <br>
ExampleInference.py展示了一个读取模型并进行推理的简单样例，并在Interface.py中调用 <br>
你可以直接运行我们提供的Test.py来获取该简易模型在SimpleData数据集上的重建结果的各项指标 <br>

## 评测指标
本次比赛主要考察大家对具包含时间空间信息的光照贴图数据的压缩，分别通过压缩率（30%），解压耗时（30%），恢复质量*40%（PSNR*20%, SSIM*10*, LPIPS*10%）三个维度进行评价。具体计算过程参考Test.py文件，会自动计算所有的评价指标。

## 提交方式
此仓库的数据是高性能算法大赛的第一个赛题，大家只考虑压缩算法迭代即可，最后将压缩文件，解压代码上传到https://github.com/TencentGemes/HRPC_Result.git 的“phase1”目录下：
1. 所有队伍先fork [HRPC_Result](https://github.com/TencentGemes/HRPC_Result.git)默认分支；
2. 大家实现压缩和解压缩算法后，提交相应文件到“phase1”的目录下，并确保Test.py文件能正常运行；
3. 每次提交需要自动判分时，大家在大赛官网上提交仓库分支，以及commit id即可。