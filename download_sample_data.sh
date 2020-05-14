function download { name="$(basename $1)"; curl "$1" > "sample_data/$name"; }
mkdir -p sample_data
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335-New2-KTK-1Perc.tar.gz'
