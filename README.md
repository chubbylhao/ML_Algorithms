<h2 align = "center">经典 Machine Learning 算法的实现</h2>

#### 有监督学习

- 线性回归：普通线性回归、Ridge回归、LASSO回归、广义线性回归
  - [机器学习中正则化项L1和L2的直观理解](https://blog.csdn.net/jinping_shi/article/details/52433975) 
  - [机器学习十大经典算法之Ridge回归和LASSO回归](https://blog.csdn.net/weixin_43374551/article/details/83688913?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166762997116782429718497%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166762997116782429718497&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-83688913-null-null.142^v63^control,201^v3^control,213^v1^t3_control2&utm_term=%E5%B2%AD%E5%9B%9E%E5%BD%92)  

- 逻辑回归：没什么特别的，只是分类作回归用罢了
  - [逻辑回归（非常详细）](https://zhuanlan.zhihu.com/p/74874291) 
- 感知机：这部分内容放到Deep Learning中可能更为合适

----------------------------以上内容很基础，没玩过也肯定听过---------------------------

-----------------------------下面的内容可能就不是那么熟悉了------------------------------

- k近邻：正所谓”近朱者赤近墨者黑“

![](https://raw.githubusercontent.com/chubbylhao/ML_Algorithms/main/supervised_learning/k_nearest_neighbors/knn.png)

- 朴素贝叶斯：自然的想法，哪类概率大，就属于哪类

![](https://raw.githubusercontent.com/chubbylhao/ML_Algorithms/main/supervised_learning/naive_bayes/naivebayes.png)

- 决策树：推荐周志华《机器学习》和李航《统计学习方法》相关部分
  - [决策树](https://chubbylhao.github.io/2022/09/25/jue-ce-shu/)

- 支持向量机：推荐李航《统计学习方法》相关部分或者戳以下链接：

  - [支持向量机通俗导论（理解SVM的三层境界）]()

- 集成学习/提升方法
  - Boosting : `AdaBoost` 
  - Bagging : `RandomForest` 

- EM

- 线性判别分析：[线性和二次判别分析](https://zhuanlan.zhihu.com/p/38641216) ~~此代码未重构

- 隐马尔科夫模型：~~略

- 条件随机场：~~略

  ------

  #### 无监督学习

- k均值聚类

- DBSCAN（密度聚类，可用于**非凸**数据）

- 奇异值分解

- 主成分分析

- Apriori

- ......

  ------

  #### 数据挖掘十大算法

———————————————————————————————————————

|                            					  	     有监督学习方法      						                             |

———————————————————————————————————————

- C4.5决策树（√）
- CART决策树（√）
- k近邻：近朱者赤近墨者黑（√）
- 朴素贝叶斯：特征独立性假设（√）
- 支持向量机：机器学习算法中的战斗机/天花板（√）
- AdaBoost：串联Boosting，三个臭皮匠顶个诸葛亮（√）
- k均值聚类：k-means（√）
- EM：是一种迭代算法，引出高斯混合模型（√）

———————————————————————————————————————

|                            					  	     无监督学习方法      						                             |

———————————————————————————————————————

- Apriori：是**关联规则**的代表性算法，可用于揭示商品之间的关联信息，从而增加销售利润（√）
- PageRank：是图的**链接分析**的代表性算法，多用于网页排序（×）

注：划√的需要重点学习，划×的可不学习

