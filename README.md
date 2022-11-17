<h2 align = "center">经典 ML 算法的实现</h2>

> 本仓库的内容将随着本人的学习而不断完善
>
> ~~持续更新中

------

#### 有监督学习

- 线性回归：普通线性回归、Ridge回归、LASSO回归、广义线性回归
  - [机器学习中正则化项L1和L2的直观理解](https://blog.csdn.net/jinping_shi/article/details/52433975) 
  - [机器学习十大经典算法之Ridge回归和LASSO回归](https://blog.csdn.net/weixin_43374551/article/details/83688913?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166762997116782429718497%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166762997116782429718497&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-83688913-null-null.142^v63^control,201^v3^control,213^v1^t3_control2&utm_term=%E5%B2%AD%E5%9B%9E%E5%BD%92)  

- 逻辑回归：实际上是线性分类模型
  - [逻辑回归（非常详细）](https://zhuanlan.zhihu.com/p/74874291) 
- 感知机：这部分内容放到Deep-Learning中可能更为合适

----------------------------以上内容很基础，没玩过也肯定听过---------------------------

-----------------------------下面的内容可能就不是那么熟悉了------------------------------

- k近邻：正所谓”近朱者赤近墨者黑“
- 朴素贝叶斯：自然的想法，哪类概率大，就属于哪类，但要注意独立性假设
- 决策树：推荐周志华《机器学习》和李航《统计学习方法》相关部分
  - [决策树（本人写的一点小总结）](https://chubbylhao.github.io/2022/09/25/jue-ce-shu/) 
- 支持向量机：推荐李航《统计学习方法》相关部分或者戳以下链接：
  - [支持向量机通俗导论（理解SVM的三层境界）](https://github.com/chubbylhao/ML_Algorithms/blob/main/supervised_learning/support_vector_machine/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%E9%80%9A%E4%BF%97%E5%AF%BC%E8%AE%BA%EF%BC%88%E7%90%86%E8%A7%A3SVM%E7%9A%84%E4%B8%89%E5%B1%82%E5%A2%83%E7%95%8C%EF%BC%89.pdf) 
- 集成学习/提升方法：正所谓”三个臭皮匠，顶个诸葛亮“，”众人拾柴火焰高“
  - Boosting :（串联）
    - AdaBoost：就像考试和错题本的关系
    - GBDT：学习残差（负梯度）的提升树
      - XGBoost：李天奇改进实现，有开源库
      - LightGBM：微软改进实现，有开源库
  - Bagging : （并联）
    - RandomForest：使用简单投票法或者简单平均法预测
- EM：最大期望算法，是一种迭代算法，常用于其它机器学习模型中（如GMM，HMM）

  - [如何通俗理解EM算法](https://blog.csdn.net/v_JULY_v/article/details/81708386) 
  - [EM——期望最大](https://zhuanlan.zhihu.com/p/78311644) 
- 线性判别分析：~~此代码未重构（线性和二次的差别仅仅在于协方差矩阵是否一致）

  - [线性和二次判别分析](https://zhuanlan.zhihu.com/p/38641216) 
  - [sklearn的官网关于线性和二次判别分析的内容](https://scikit-learn.org/stable/modules/lda_qda.html) 
- 隐马尔科夫模型：~~略
- 条件随机场：~~略
- ......

------

#### 无监督学习

- k均值聚类：也许这是人们听过最多的也是最基本的聚类方法了
- DBSCAN：密度聚类，相比于k均值，其可用于非凸数据样本
- 高斯混合模型：任意多高斯模型的线性组合理论上可以表示所有类型的概率分布模型

  - [高斯混合模型及其EM算法的理解](https://blog.csdn.net/jinping_shi/article/details/59613054) 
  - [详解EM算法和高斯混合模型](https://blog.csdn.net/lin_limin/article/details/81048411?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166771942416800182189305%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166771942416800182189305&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-3-81048411-null-null.142^v63^control,201^v3^control,213^v1^t3_control2&utm_term=%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B) 

![](https://raw.githubusercontent.com/chubbylhao/ML_Algorithms/main/unsupervised_learning/clustering.png)

> BRICH、STING和谱聚类目前于我而言尚有一点难度，只能以后需要用到的时候再回来学习了

- 奇异值分解SVD：降维压缩方法
- 主成分分析PCA：降维压缩方法
- 核主成分分析KPCA：通过核技巧使PCA具有非线性特性
- MDS：使用欧氏距离时与PCA等价（保持了欧氏距离）
- Isomap：使用测地距离代替MDS中的欧氏距离（保持了测地距离）
- LLE：保持了样本和邻域之间的线性关系

以下为几种经典的降维算法分别在不同数据上的处理结果：

|                           降维方法                           |                          iris 数据                           |                         s curve 数据                         |                       swiss roll 数据                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|         多维缩放 (MDS, Multiple Dimensional Scaling)         | <img src="https://raw.githubusercontent.com/chubbylhao/ML_Algorithms/main/unsupervised_learning/MDS/mds_iris.png" style="zoom: 67%;" /> | <img src="https://raw.githubusercontent.com/chubbylhao/ML_Algorithms/main/unsupervised_learning/MDS/mds_s_curve.png" style="zoom: 67%;" /> | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/MDS/mds_swiss_roll.png?raw=true" style="zoom: 67%;" /> |
|        主成分分析 (PCA, Principal Component Analysis)        | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/PCA/pca_iris.png?raw=true" style="zoom: 67%;" /> | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/PCA/pca_s_curve.png?raw=true" style="zoom: 67%;" /> | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/PCA/pca_swiss_roll.png?raw=true" style="zoom: 67%;" /> |
| 核主成分分析 (KPCA, Kernelized Principal Component Analysis) |                                                              | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/KPCA/kpca_s_curve.png?raw=true" style="zoom: 67%;" /> | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/KPCA/kpca_swiss_roll.png?raw=true" style="zoom: 67%;" /> |
|            等度量映射 (Isomap, Isometric Mapping)            |                                                              | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/Isomap/isomap_s_curve.png?raw=true" style="zoom:67%;" /> | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/Isomap/isomap_swiss_roll.png?raw=true" style="zoom:67%;" /> |
|         局部线性嵌入 (LLE, Locally Linear Embedding)         |                                                              | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/LLE/lle_s_curve.png?raw=true" style="zoom:67%;" /> | <img src="https://github.com/chubbylhao/ML_Algorithms/blob/main/unsupervised_learning/LLE/lle_swiss_roll.png?raw=true" style="zoom:67%;" /> |

- Apriori：关联规则分析，实际上就是概率与组合的问题

- ......

  ------


#### 数据挖掘十大算法

- C4.5决策树（√）
- CART决策树（√）
- k近邻：近朱者赤近墨者黑（√）
- 朴素贝叶斯：特征独立性假设（√）
- 支持向量机：机器学习算法中的战斗机/天花板（√）
- AdaBoost：串联Boosting，三个臭皮匠顶个诸葛亮（√）
- k均值聚类：k-means（√）
- EM：是一种迭代算法，可用于高斯混合模型（√）
- Apriori：是**关联规则**的代表性算法，可用于揭示商品之间的关联信息，从而增加销售利润（√）
- PageRank：是图的**链接分析**的代表性算法，多用于网页排序（×）

注：划√的需要重点学习，划×的可不学习

------

#### 补充说明

对于机器视觉方向来说，以上内容基本足够，至于什么概率图啊啥啥啥的，等以后涉及到或者有闲暇工夫的时候再深入学习吧~~

现在花大力气去学习语音识别、自然语言处理、推荐系统啥的属实没有必要~~

------

#### 资源列表

- [ShowMeAI](https://www.showmeai.tech/) 
- [scikit-learn](https://scikit-learn.org/stable/index.html) 

