、
  

先上论文地址
1.
[Bandyopadhyay S, Lokesh N, Murty M N. Outlier aware network embedding for attributed networks\[C\]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 12-19.](https://link.zhihu.com/?target=https%3A//www.aaai.org/ojs/index.php/AAAI/article/download/3763/3641)
2.
[Bandyopadhyay S, Vivek S V, Murty M N. Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding\[C\]//Proceedings of the 13th International Conference on Web Search and Data Mining. 2020: 25-33.](https://link.zhihu.com/?target=https%3A//dl.acm.org/doi/abs/10.1145/3336191.3371788)

### 1.背景介绍

现有的属性图上节点的embedding研究有了很大进展，往往利用深度学习，流行的gcn获得了很大的突破，但是如果属性图本身有异常点，这些异常点会严重影响最后的embedding学习结果。本文就是在属性图上学习异常点感知的embedding表示。

### 2.异常定义

首先，这篇文章定义了什么样子是异常（我觉得这一步是论文后续发展的关键，定义好一个合理异常，才有后面的工作）

![](./attachments/1633238061193.jpg)

同一种颜色属于一个community。正常情况下一个commnuity的团体在属性和结构上更加相似（紧密）。 异常a（结构异常）:属性上属于红色的community（正常），结构上和其它community的节点有连接。 异常b（属性异常）:结构上属于红色的community（正常），属性上和其它community的属性比较相似，也就是说在 只从属性上看， 这个节点并不完全遵从红色社区的模式。 异常c（综合异常）:节点上属于绿色社区，属性上属于蓝色社区。也就是在 属性上和结构上都存在异常。

### 3.问题定义

G = \(V,E,C\), where V = \{v1,v2,··· ,vN\} ，A是邻接矩阵\(N_N\)，C是内容矩阵（N_K）, 图上的研究问题的一个通病A是非常稀疏的。（有些异常检测方法针对稀疏性换模型）最终获得节点的低维表示。 这是个无监督下的异常检测问题。 
模型 下面无论内容还是结构，都是基于重构损失的。

## 4模型

**4.1 Learning from the Link Structure**

![](./attachments/1633238061199.jpg)

A是邻接矩阵。G是低维表示，即我们想要最终学习到的。H是个变换矩阵。 动机是这样的：目前表示学习获得的隐向量，损失函数是基于重构损失，也就是上面的式子平方项。但是因为这种学习只能够记住大部分的模式，异常点的定义本身就是与其它点“differ significant from others”，异常点可能对这个损失做出了很大的贡献，因此作者的想法就是 让异常点 对最终的损失做出较小的贡献。极端情况下我们擦除这些异常点，不就是一个很好的表示学习过程了吗？

**4.2 Learning from the Attributes**

![](./attachments/1633238061200.jpg)

一样的想法

**4.3 Connecting Structure and Attributes** 上面是单独考虑结构或者属性。我理解 上面2个损失函数可以分别对a异常和b异常做出很好的结果。针对第三种异常，联合异常。如果单看属性，不存在异常。如果单看结构，不存在异常。同时考虑就存在异常了。因此：对结构特征和属性特征做约束，因为是一个节点的不同表示，具有约束性。

![](./attachments/1633238061201.jpg)

G和U可能不对齐。因此 Embedding Transformation and Procrustes problem 乘了一个带约束的正交矩阵做 **特征对齐！！！** 也就是 Procrustes problem（百度了一下经常用于图像领域的pixel 对齐）. 个人理解对W不做约束也行，后面就没闭式解了，估计作者也试过\~

![](./attachments/1633238061201.jpg)

**4.4 联合损失**

![](./attachments/1633238061202.jpg)

**4.5 更新： 固定其它参数，一次更新一个。 4.6 Updating G, H, U, V G的更新：**

![](./attachments/1633238061233.jpg)

求个导数=0，高等代数基础就可以推出来。 其它类似

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061304.jpg)

**4.7 Updating W**

![](./attachments/1633238061271.jpg)

转换成了 Procrustes problem，直接闭式解。 4.8 Updating O 带有等式约束的拉格朗日问题。

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061304.jpg)

高等代数知识求解\~ 4.9 Algorithm:ONE

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061314.jpg)

  

### 5. 实验！！！！

初始化：G和U用矩阵分解得到的表示。 
**数据集**

![](./attachments/1633238061272.jpg)

**异常点设置方式：** 1.计算每类节点的class 分布 2.根据概率选择一类节点当作结构上的异常 3.在这类节点种plant一个节点，让这个节点（m+10\%）的边和别的类别相连。m是这个类别下的平均度数。 4.内容和这个类别语义上相同。（the content of the structural outlier node is made semantically consistent with the keywords sampled from the nodes of the selected class）

Outlier Detection， Node Classiﬁcation Node Clustering

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061315.jpg)

AANE算法有时候和它相当

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061316.jpg)

![](./attachments/1633238061274.jpg)

SEANO是个半监督，但是没有这篇文章的好。但是AANE在citeseer数据集表现更好。 后面要看下这个算法。

### 6 总结和贡献：！

1.第一篇 异常点敏感的embedding学习 
2.用了Procrustes problem解决对齐问题。 
3.实验表现很好

## 7.后续贡献
接下来作者继续做了一些工作发在了wasm上。

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061279.jpg)

作者分别提出了2个算法Done和Adone 贡献点 1.用一个AE的隐向量代替了G和U。 2.用判别器去代替Procrustes problem，通过最小化判别器的损失，让2个AE生成出来的隐向量维度对齐\!\!很聪明\!\!\!

实验： 异常检测任务：

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061280.jpg)

dominant效果很好，是个半监督的方法，后面要看下\~

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061315.jpg)

有异常的时候Adone就很好\~ 节点分类

![](https://raw.githubusercontent.com/chenbofeng123/blog_pic/master/小书匠/1633238061317.jpg)

验证adversial的有效性

![](./attachments/1633238061272.jpg)