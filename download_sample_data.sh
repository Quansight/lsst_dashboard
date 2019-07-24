function download { name="$(basename $1)"; curl "$1" > "sample_data/$name"; }
mkdir -p sample_data
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst_sampled_1000/HSC-G.h5'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst_sampled_1000/HSC-I.h5'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst_sampled_1000/HSC-R.h5'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst_sampled_1000/HSC-Z.h5'
