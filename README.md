# Heritage BIM Registration (动态建模插件 SemRegPy)

## Introduction

本插件以 Xue 等 (2019) 提出的语义配准 (Semantic Registration) 技术为基础，针对《面向粵港澳歷史文化保護傳承的虛擬現實技術研究與應用》(The applications of Virtual Reality technologies for cultural heritage conservation in the Guangdong-Hong Kong-Macao Greater Bay Area)项目而特别研发。

通过 semregpy.core, semregpy.fitness, 和 semregpy.component 等 Python 接口，各类 GIS/BIM API 平台（例如 ArcGIS Pro 和 Revit）和应用提供动态建模的功能。（接口的参数类型和调用顺序，以本演示程序的内嵌流程为例）

Xue, F., Lu, W., Chen, K., & Zetkulic, A. (2019). [From semantic segmentation to semantic registration: Derivative-Free Optimization–based approach for automatic generation of semantically rich as-built Building Information Models from 3D point clouds. Journal of Computing in Civil Engineering](https://doi.org/10.1061/(ASCE)CP.1943-5487.0000839), 33(4), 04019024.

Algorithm: [@Fan Xue](https://github.com/ffxue), [@Siyuan Meng](https://www.researchgate.net/profile/Siyuan-Meng-6)

UI: [@Longyong Wu](https://www.github.com/chunibyo-wly)

![Demo](https://raw.githubusercontent.com/chunibyo-wly/image-storage/master/202412212232806.png)

https://github.com/user-attachments/assets/2e57a01f-6519-47b8-9899-3d9f7379fd88


## How to install

| Package       | Version              |
|---------------|----------------------|
| CloudCompare  | v2.13.2              |
| Python        | CloudCompare Bundled |
| open3d        | 0.18.0               |
| nlopt         | 2.8.0                |
| psutil        | 6.1.0                |
| customtkinter | 5.2.2                |
| numpy         | 1.26.4               |

![](https://raw.githubusercontent.com/chunibyo-wly/image-storage/master/202501062140468.png)
![](https://raw.githubusercontent.com/chunibyo-wly/image-storage/master/202501062140451.png)