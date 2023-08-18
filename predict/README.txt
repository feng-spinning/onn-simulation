welcome! 在开始预测之前，请先阅读项目文档，这样你才能了解本部分"预测"的具体意义。

使用源代码进行预测（推荐）：

step1: 安装依赖

相信读这份文档的人一定都安装好了 `torch`, `numpy` & `matplotlib`。作者使用的环境是：

torch 2.0.1+cpu（训练使用的是cuda，预测cpu or cuda均可）

matplotlib 3.7.1

numpy 1.23.5，更高版本也可兼容。

step2: 预处理

运行prepro.py，建议只取前 2% 的数据，仅作展示。

step3：进行预测

最终本代码会显示一张对比图，分别是入射光场和最后的成像光强分布。图的示例在 combined_figure.png 中。

对于只有相位调制的模型，本代码提供了训练好的两个权重：分别为`weights_large2.pt` & `weights_small3.pt`。

可以在line 128更改

model.load_state_dict(torch.load("weights_large2.pt",map_location=torch.device(device)))

为你所需要的权重。

对于有相位+振幅调制的模型，本代码提供了训练好的一个权重：`weights_large_am.pt`。

对于引入relu函数的模型，本代码提供了一个训练好的权重：`weights_large_relu.pt`

好啦，这就是需要进行预测的全部步骤啦！祝你玩的开心！

--------------------
作者：未央-精21 冯子嘉
联系方式：fengzj22@mails.tsinghua.edu.cn