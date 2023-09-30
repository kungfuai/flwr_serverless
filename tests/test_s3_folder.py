"""
Use moto to mock S3
"""
from pytest import raises
import boto3
from moto import mock_s3
from flwr_serverless.shared_folder.s3_folder import (
    S3FolderWithBytes,
    S3FolderWithPickle,
)


def test_s3_bytes_folder_read_write_delete():
    with mock_s3():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="test_bucket")
        folder = S3FolderWithBytes(
            "test_bucket/test_folder",
            retry_sleep_time=0.1,
            max_retry=10,
        )
        # read and write a dummy file
        key = "dummy"
        folder[key] = b"dummy"
        assert folder[key] == b"dummy"
        del folder[key]


def test_s3_pickle_folder_read_write_delete():
    with mock_s3():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="test_bucket")
        folder = S3FolderWithPickle(
            "test_bucket/test_folder",
            retry_sleep_time=0.1,
            max_retry=10,
        )
        # read and write a dummy file
        key = "dummy"
        folder[key] = "dummy"
        assert folder[key] == "dummy"
        del folder[key]


def test_when_s3_is_not_accessible():
    with mock_s3():
        s3 = boto3.client("s3")
        S3FolderWithBytes(
            "test_bucket/test_folder",
            retry_sleep_time=0.1,
            max_retry=10,
            check_at_init=False,
        )
        with raises(s3.exceptions.NoSuchBucket):
            S3FolderWithBytes(
                "test_bucket/test_folder",
                retry_sleep_time=0.1,
                max_retry=10,
                check_at_init=True,
            )
