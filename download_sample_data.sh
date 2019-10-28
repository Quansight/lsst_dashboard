function download { name="$(basename $1)"; curl "$1" > "sample_data/$name"; }
mkdir -p sample_data
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335/metadata.yaml'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335/qaDashboardCoaddTable-9813.parq'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335/qaDashboardVisitTable-9813.parq'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335/qaDashboardCoaddTable-9697.parq'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335/qaDashboardVisitTable-9697.parq'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335/qaDashboardCoaddTable-9615.parq'
download 'https://quansight.nyc3.digitaloceanspaces.com/datasets/lsst/DM-21335/qaDashboardVisitTable-9615.parq'