# Graph-to-Tree Learning for Solving Math Word Problems

PyTorch implementation of Graph based Math Word Problem solver described in our ACL 2020 paper Graph-to-Tree Learning for Solving Math Word Problems. In this work, we propose a solution for Math Word Problem Solving via graph neural network.

## Steps to run the experiments

### Requirements
* ``Python 3.6 ``
* ``>= PyTorch 1.0.0``

For more details, please refer to requiremnt file.

### Training
#### [MATH23K]
first get into the math23k directory:
* ``cd math23k``

training-test setting :
* ``python run_seq2tree_graph.py``

cross-validation setting :
* ``python cross_valid_graph2tree.py``

#### [MAWPS]
cross-validation setting :
* ``cd mawps``
* ``python cross_valid_mawps.py``

### Reference
```
@article{zhang2020graph2tree,
  title={Graph-to-Tree Learning for Solving Math Word Problems},
  author={Zhang, Jipeng and Wang, Lei and Lee, Roy Ka-Wei and Bin, Yi and Shao, Jie and Lim, Ee-Peng},
  journal={ACL 2020},
  year={2020}
}
```
