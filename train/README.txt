文件解释：

step1: 安装依赖

相信读这份文档的人一定都安装好了 `torch`, `numpy`。作者使用的环境是：

torch 2.0.1+cpu（训练使用的是cuda，预测cpu or cuda均可）

numpy 1.23.5，更高版本也可兼容。

step2：预处理（可选）

使用prepro.py以及prepro_label.py，安装文件提示进行更改，保存npy文件即可完成。预处理应该生成6个npy文件。

step3：训练

训练：共有四种训练模型，都以train开头。其中train_small.py可以不经预处理直接运行，其余均为大数据集训练，需要走预处理步骤。

推荐使用GPU进行预测，加快速度。cpu亦可。

训练完成后会将权重保存至一个pt文件中，将pt文件复制到model文件夹中即可开始使用该模型进行预测。

-------------------
作者：未央-精21 冯子嘉
联系方式：fengzj22@mails.tsinghua.edu.cn