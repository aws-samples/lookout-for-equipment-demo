# Automate detection of abnormal equipment behavior and review predictions with human in the loop using Amazon Lookout for Equipment and Amazon A2I
---
In this blog we will show you how you can setup Amazon Lookout for Equipment to train an abnormal behavior detection model using a wind turbine dataset for predictive maintenance and setup up a human in the loop workflow to review the predictions using Amazon A2I, augment the dataset and retrain the model. To get started with Amazon Lookout for Equipment, we will create a dataset, ingest data, train a model and run inference by setting up a scheduler. After going through
these steps we will show you how you can quickly setup human review process using Amazon A2I and retrain your model with augmented or human reviewed datasets.

We will walk you through the following steps: 
1. Creating a dataset in Amazon Lookout for Equipment
2. Ingesting data into the Amazon Lookout for Equipment dataset
3. Training a model in Amazon Lookout for Equipment
4. Running diagnostics on the trained model
5. Creating an inference scheduler in Amazon Lookout for Equipment to send a simulated stream of real-time requests
6. Setting up an Amazon A2I private human loop and reviewing the predictions from Amazon Lookout for Equipment
7. Retraining your Amazon Lookout for Equipment model based on augmented datasets from Amazon A2I

Please follow the instructions in the [accompanying blog post](aws.amazon.com) to clone this repository and get started with your Jupyter notebook
