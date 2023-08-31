from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="flwr_serverless",
    version="0.1",
    description="A peer2peer federated learning library based on flwr",
    url="https://github.com/kungfuai/flwr_serverless",
    author="Kungfu AI",
    author_email="zhangzhang.si@gmail.com",
    license="MIT",
    packages=["flwr_serverless"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
