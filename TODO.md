- keras: scale to 3+ clients
- test real concurrency
- s3 storage backend


Reese TODO:
- add callback to non_federated and federated that uploads the non-partioned accuracy to W&Bs?
    - currently looks confusing as it has high accuracy on partition but not on whole dataset
- experiment with various configurations
- make sure all 3 types of async federated leraning work, as well as sync
