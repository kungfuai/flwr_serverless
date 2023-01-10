- keras: scale to 3+ clients
- test real concurrency
- s3 storage backend


Reese TODO:
- ask ZZ if he wants to use a validation sets to test accuracy per step

- Make OO experimental model setups
    - parent experimental_model class
        - takes config and dataset
        - has train_and_eval() method for easy expermtianl tracking on W&Bs
    - federal_learning_model class for testing various federal learning methods

- get basic scripts working

- add callback to non_federated and federated that uploads the non-partioned accuracy to W&Bs
    - currently looks confusing as it has high accuracy on partition but not on whole dataset
