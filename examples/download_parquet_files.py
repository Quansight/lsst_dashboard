import boto3
import os

S3_LOCATION = "http://{}.s3.amazonaws.com/".format("lsst-sample-data")
BUCKET_NAME = "lsst-sample-data"


def download_from_s3(remote_directory_name):

    print("downloading {}".format(remote_directory_name))

    s3_resource = boto3.resource("s3")

    bucket = s3_resource.Bucket(BUCKET_NAME)
    for object in bucket.objects.filter(Prefix=remote_directory_name):
        if not os.path.exists(os.path.dirname(object.key)):
            os.makedirs(os.path.dirname(object.key))
        bucket.download_file(object.key, object.key)


download_from_s3("HSC-I_visits")
download_from_s3("HSC-R_visits")
download_from_s3("HSC-Z_visits")
download_from_s3("analysisCoaddTable_forced")
download_from_s3("analysisCoaddTable_unforced")
