文件解释：（updated 2024.4.23，涉及预处理，一定要看！）
step1: 安装依赖

相信读这份文档的人一定都安装好了 `torch`, `numpy`。作者使用的环境是：
torch 2.0.1+cpu（训练使用的是cuda，预测cpu or cuda均可）
numpy 1.23.5，更高版本也可兼容。

step2：预处理
使用上一级目录下的prepro.py直接得到处理过后的数据。注意：prepro.py中有一个ratio参数，决定了你用MNIST中的多少比例进行训练。

step3：训练
训练：共有四种训练模型，都以train开头。建议直接运行train_main.py，是最简单的一种神经网络，只有相位调制，没有非线性。
推荐使用GPU进行预测，加快速度。cpu亦可。
训练完成后会将权重保存至一个pt文件中，将pt文件复制到model文件夹中即可开始使用该模型进行预测。

-------------------
作者：未央-精21 冯子嘉
联系方式：fengzj22@mails.tsinghua.edu.cn
