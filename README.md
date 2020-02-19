# POlicy optimizaTION

How to install (tested on Ubuntu 18.04.2):

```bash
apt-get update
apt-get install git python3 python3-pip
git clone https://github.com/T3p/potion.git
cd potion
pip3 install -e .
```

How to execute the experiments (tested with the values reported in Appendix D): 

```
cd potion
cd DPO
python main.py --task "name"
```
where "name" can be:
- 'mini' to run the minigolf task);
- 'mass' to run the double integrator task,
- 'safe' to run the robot adaptation task.

These are the optional parameter that can be specified (default values are the ones used in the final experiments reported in the paper):
- --seed: value of the initial seed, int;
- --nruns: n. of runs of the task, int;
- --alpha: learning rate used in policy projection, float;
- --lambda: regularization used in policy projection, float;
- --mcrst: number of abstract states on each state dimension, int;
- --batch: size of the batch, int;
- --nsteps: size of an episode, int;
- --niter: iterations of the algorithm, int;
- --Lds: lipschitz constant to be used in the minigolf task, float.
