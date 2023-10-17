FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /workspace
RUN pip install --upgrade pip && \
    pip install flwr==1.5.* keras-cv==0.6.* python-dotenv wandb

