## Installation
You can install BestMSM by simply downloading the package from the GitHub
repository and using the standard installation instructions for packages built
using [Distutils](https://docs.python.org/2/distutils/index.html)

```
git clone http://github.com/daviddesancho/bestmsm destination/bestmsm
cd destination/bestmsm
python setup.py install --user
```
### Parallel processing in Python and BestMSM
In BestMSM we make ample use of the multiprocessing library, which for 
MacOS X can conflict with non-Python libraries. This is a problem that
can result in segmentation faults. Digging in the internet I found a 
workaround for this problem, by setting the following environment 
variable

```
export VECLIB_MAXIMUM_THREADS=1
```

This should be set in the terminal before you start your python session.
