
## LSST Data Processing Explorer
[![Travis Build Status](https://travis-ci.com/Quansight/lsst_dashboard.svg?branch=master)](https://travis-ci.com/Quansight/lsst_dashboard)  
  
### Getting Started...

1. Download and Extract sample dataset here:
https://quansight.nyc3.digitaloceanspaces.com/datasets/RC2_w18.tar.gz

2. Setup and activate conda environment with Python 3.7 and PyViz stack
```
conda env create -f environment.yml
source activate lsst-panel
conda install scikit-learn
python setup.py develop
```

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

### For developers...
To increment the version number, run `rever <version>` 
where version is the new version number. This will overwrite
the version in the places define in `VERSION_BUMP_PATTERNS`:  
```
$VERSION_BUMP_PATTERNS = [  # These note where/how to find the version numbers
                         ('lsst_dashboard/__init__.py', '__version__\s*=.*', "__version__ = '$VERSION'"),
                         ('setup.py', 'version\s*=.*,', "version='$VERSION',")
                         ]
```

This will also update .authors.yml, AUTHORS.md, .mailmap, with accurate data from the repository
