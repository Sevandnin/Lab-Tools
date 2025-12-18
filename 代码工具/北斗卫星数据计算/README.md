# 北斗卫星数据预处理工具 (BDS Data Preprocessing Tool)

## 功能介绍

本工具用于处理北斗卫星导航系统的观测数据，将原始RINEX格式的观测文件和导航文件转换为可用于后续处理的标准化数据格式。

主要功能包括：

1. 解析RINEX 3.x格式的观测文件和导航文件
2. 计算卫星位置和速度
3. 进行卫星轨道解算
4. 计算电离层延迟修正
5. 计算对流层延迟修正
6. 坐标系统转换（ECEF <-> 地理坐标）
7. 输出处理后的标准化数据到Excel文件

## 输入参数

- **观测文件(obs)**: RINEX格式的北斗观测数据文件
- **导航文件(nav)**: RINEX格式的北斗导航电文文件
- **接收机近似位置**: 支持三种输入方式：
  - ECEF坐标(X,Y,Z)
  - 地理坐标(纬度,经度,高度)
  - 从RTK数据文件(.xlsx)中读取第一行位置信息
- **电离层参数Alpha和Beta**: 用于Klobuchar电离层模型的参数，以逗号分隔输入
- **年积日(DOY)**: 用于大气延迟计算

## 输出数据

输出文件为`BDSData_output.xlsx`，包含以下列：

1. Epoch: 历元编号
2. GPS_Week: GPS周数
3. GPS_TOW: GPS周内秒
4. Sat_PRN: 卫星PRN号
5. Pseudorange_Corrected: 修正后的伪距测量值
6. Sat_X:卫星ECEF坐标X
7. Sat_Y:卫星ECEF坐标Y
8. Sat_Z: 卫星ECEF坐标Z
9. Pseudorange_Rate: 伪距率
10. Sat_Vel_X: 卫星X轴速度
11. Sat_Vel_Y: 卫星Y轴速度
12. Sat_Vel_Z: 卫星Z轴速度
13. SNR: 信噪比
14. Elevation: 卫星高度角
15. Azimuth: 卫星方位角

## 使用方法

运行[bds_data_preprocess.py](file:///e:/Desktop/Lab/Tools/%E4%BB%A3%E7%A0%81%E5%B7%A5%E5%85%B7/%E5%8C%97%E6%96%97%E5%8D%AB%E6%98%9F%E6%95%B0%E6%8D%AE%E8%AE%A1%E7%AE%97/bds_data_preprocess.py)脚本，按照提示输入相应参数即可。

## 更新日志

### 2025-12-18 bds_data_preprocess_v2
- 添加输入数据复述功能，在处理前显示所有输入参数供用户检查
- 支持经纬高(LLH)坐标输入，程序会自动转换为ECEF坐标
- 增加坐标转换函数[blh_to_xyz](file:///e:/Desktop/Lab/Tools/%E4%BB%A3%E7%A0%81%E5%B7%A5%E5%85%B7/%E5%8C%97%E6%96%97%E5%8D%AB%E6%98%9F%E6%95%B0%E6%8D%AE%E8%AE%A1%E7%AE%97/bds_data_preprocess.py#L128-L161)，完善坐标系统转换功能
- 新增支持从RTK数据文件(.xlsx)中读取接收机位置信息功能
- 修改电离层参数输入方式，改为逗号分隔

### [历史更新记录待补充]

## 技术细节

该工具实现了完整的北斗卫星数据预处理流程，包括：

- RINEX文件解析器（支持版本3.x）
- 卫星轨道解算（基于开普勒方程）
- 相对论效应修正
- 电离层延迟修正（Klobuchar模型）
- 对流层延迟修正（EGNOS模型和Neill映射函数）
- 坐标系统变换（ECEF <-> 地理坐标）

## 依赖库

- numpy
- pandas
- datetime
- math
- os
- tkinter (可选，用于文件选择对话框)

## 注意事项

1. 输入的RINEX文件应为同一时间段的数据
2. 确保输入的接收机位置尽可能准确，以获得更好的卫星仰角和方位角计算结果
3. 电离层参数可以从IGS等机构获取，对于高精度应用建议使用实时或后处理参数
4. RTK数据文件需符合格式要求：第一列为周内秒，第二列为纬度，第三列为经度，第四列为高程