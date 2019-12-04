function download { name="$(basename $1)"; curl "$1" > "sample_data/$name"; }
mkdir -p sample_data
# small test dataset 1 percent of full (approx 750mb)
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335-1Perc.tar.gz'
# larger test dataset 10 percent of full (approx 6GB)
# download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335-10Perc.tar.gz'

# data needs to be untarred.