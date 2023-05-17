#### Run Code:

python main_fed.py   --gpu 0 --local_ep 5 --local_bs 10  --num_users 2048 --frac 1 --dataset [fashionmnist,cifar,agnews]  --model [cnn,resnet,lstm] --epochs 250 [--fedslice] [--iid] [--loc 50] [--scale 25] [--dataset_size 50] 

num_users represents the number of edge devices

- **'num_users'**  represents the total number of edge devices.
- **'frac'** represents  the proportion of edge devices participating in the training each epoch.
- This parameter **'fedslice'** means that the FedSlice algorithm is used; without this parameter, the FedAvg algorithm is used.
- **'dataset_size'** denotes the training size of each client  when **'--iid'** is used .
- **'loc'** and **'scale'** denote the expectation and standard deviation of the normal distribution when **'--iid'** is not used .
- Other detailed parameter settings can be obtained from file **'./utils/options.py'**

---

#### Note:

#### Our code implementation only simulates the federated learning environment.

#### Code implementation of the main Federated Learning structure modified from:

Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561
