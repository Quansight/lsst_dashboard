
## LSST Data Processing Explorer

### Getting Started...

1. Download and Extract sample dataset here:
https://quansight.nyc3.digitaloceanspaces.com/datasets/RC2_w18.tar.gz

2. Setup and activate conda environment with Python 3.7 and PyViz stack

3. Use panel off v0.7.0a2 branch (temporarily until official release)
```
git clone https://github.com/pyviz/panel
cd panel
git checkout v0.7.0a2
source activate my-lsst-conda-env
python setup.py develop

# restart server...
```

4. To launch dashboard in dev mode (server will restart, whenever files change):

`LSST_SAMPLE_DATA=/home/dharhas/RC2_w18/ panel serve dashboard.py --dev lsst_dashboard --show --port 5005`