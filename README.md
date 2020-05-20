## LSST Data Processing Explorer

Here are end-to-end instructions of how to currently set up and run the in-development QA Dashboard on a dataset of your choosing.  N.B. the dataset must have had the `pipe_analysis` scripts and `visitMatch.py` run on them already.

### Setting up the stack

After sourcing the current stack, set up the `tickets/DM-21335` branches of `qa_explorer` and `obs_base`.  Run `scons` on `qa_explorer` in order to make the `prepareQADashboard.py` command-line task available.

### Install this package and various dependencies

After the stack has been set up, install this package and a number of dependencies:
```
git clone https://github.com/quansight/lsst_dashboard

cd lsst_dashboard
pip install --user requirements.txt
pip install --user -e .
```

### Run `prepareQADashboard.py`

Run the `prepareQADashboard.py` script on a repo as follows:

```
prepareQADashboard.py /datasets/hsc/repo/rerun/RC/w_2020_14/DM-24359 --output /my/dashboard/data/path/w_14 --id tract=9615^9697^9813  filter=HSC-G^HSC-R^HSC-I^HSC-Z^HSC-Y^NB0921 --clobber-config --no-versions
```

This is a very lightweight task that basically just writes the valid dataIds to a `.yaml` file that the data-repartitioning step will read.

### Repartition dataset

In order to read the on-disk data with maximal efficiency, we write the necessary data to [kartothek](https://kartothek.readthedocs.io/en/latest/) datasets, using the `lsst_data_repartition` command-line interface, pointing to the *output* repo of the `prepareQADashboard.py` task:

```
lsst_data_repartition --queue=normal --nodes=4 /my/dashboard/data/path/w_14
```

This program launches a dask cluster via slurm and uses that to manage the data repartitioning, which by default writes a new directory called `ktk` underneath the repo path entered above.  You can follow the progress of the dask tasks by opening a tunnel to the dask dashboard port (usually port `20000`), e.g.

```
ssh -NfL localhost:20000:localhost:20000 lsst-dev
```

and pointing a browser to `localhost:20000`.  The number of nodes (and various available chunking options) you want will depend on the size of the dataset you are repartitioning.  For datasets on the scale of RC2, defaults should generally be fine.

### Launch dashboard

If the above repartitioning completes correctly, you can then launch the dashboard and point it to that dataset.  You will also have to tunnel ports, as before; the default dashboard port is `20500`, but see the command-line output to make sure:

```
lsst_data_explorer --queue=normal --nodes=4
```

When the dashboard starts, point your browser to the correct address, enter the path to the kartothek-repartitioned dataset (e.g., `/my/dashboard/data/path/w_14/ktk`) in the box in the top-right of the window, and click the "Load Data" button.  It's good to also have another window pointing to the dask dashboard, so you can see dask activity when it happens.


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
