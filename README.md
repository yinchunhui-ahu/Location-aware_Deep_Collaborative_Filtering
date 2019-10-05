
# Location-aware Deep Collaborative Filtering

This is the implementation of our paper:

Yiwen Zhang, Chunhui Yin, Qilin Wu*, Qiang He and Haibin Zhu. [Location-aware Deep Collaborative Filtering for Service Recommendation](https://ieeexplore.ieee.org/document/8805172), IEEE Transactions on Systems, Man, and Cybernetics: Systems. 10.1109/TSMC.2019.2931723, 2019. (SCI)

Developer: Chun-hui Yin

Affiliate: [Big Data and Cloud Service Lab](http://bigdata.ahu.edu.cn), Anhui University

Last updated: 2019/10/05

**Please cite our paper if you use our codes. Thanks!** 

# Environment Requirement

This code can be run at following requirement but not limit to:
- python = 3.6.6
- keras = 2.0.9
- pandas = 0.23.4
- numpy = 1.14.0
- scikit-learn = 0.21
- other installation dependencies required above

# Example of Usage

&gt;&gt;&gt;python run_rt.py

&gt;&gt;&gt;python run_tp.py

# Dataset

- To simulate the real-world situation, we sparse the original matrix at six densities and generate instances for training
- Here we provide the preprocessed real-world dataset WS-Dream (dataset#1)

# Note

- Experiments can be run on multi-core CPUs at 6 densities by turning on parallel mode
