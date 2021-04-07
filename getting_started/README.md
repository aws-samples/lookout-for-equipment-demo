# Amazon Lookout for Equipment Demo
Amazon Lookout for Equipment uses the data from your sensors to detect abnormal equipment behavior, so you can take action before machine failures occur and avoid unplanned downtime.

**Note:** *Expect around 1.5 hour to run this whole set of notebooks, including 55 minutes of training time.*

## Overview
Amazon Lookout for Equipment analyzes the data from your sensors, such as pressure, flow rate, RPMs, temperature, and power to automatically train a specific ML model based on just your data, for your equipment – with no ML expertise required. Lookout for Equipment uses your unique ML model to analyze incoming sensor data in real-time and accurately identify early warning signs that could lead to machine failures. This means you can detect equipment abnormalities with speed and precision, quickly diagnose issues, take action to reduce expensive downtime, and reduce false alerts.

### Installation instructions
[**Create an AWS account**](https://portal.aws.amazon.com/gp/aws/developer/registration/index.html) if you do not already have one. This service is actually in Preview: to run this demo, your AWS account will need to be on the allow list. To sign up for the preview, you can follow [**this link**](https://pages.awscloud.com/Amazon-Lookout-for-Equipment-Preview.html).

Once you have access to the service, login to your AWS account and navigate to the SageMaker console and create a new instance. Using an **ml.m5.xlarge instance** with the standard 5 GB attached EBS volume is recommended to process and visualize the dataset comfortably. To enable exploration of big timeseries dataset, you might need to increase the EBS volume size. Some plots can take up a significant amount of memory: in such exploration, it's not unusual to move to bigger memory optimized instance (like the **ml.m5.4xlarge** one).

You need to ensure that this notebook instance has an **IAM role** which allows it to call the Amazon Lookout for Equipment APIs:

1. In your **IAM console**, look for the SageMaker execution role endorsed by your notebook instance (a role with a name like `AmazonSageMaker-ExecutionRole-yyyymmddTHHMMSS`)
2. On the `Permissions` tab click on `Add inline policy`, select `JSON` and enter the following policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "lookoutequipment:*",
            "Resource": "*"
        },
        {
            "Action": "iam:PassRole",
            "Effect": "Allow",
            "Resource": "*"
        }
    ]
}
```
3. Click on `Review policy` and name it (e.g. **LookoutEquipmentAccess**) before clicking on `Create policy`
4. Browse to the `Trust relationship` tab for this role, click on the `Edit trust relationship` button and fill in the following policy. You may already have a trust relationship in place for this role, in this case, just add the **"lookoutequipment.amazonaws.com"** in the service list:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "sagemaker.amazonaws.com",
          "lookoutequipment.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```
5. Click on `Update the Trust Policy`: your SageMaker notebook instance can now call the Lookout for Equipment APIs and the service will have the appropriate access to the S3 buckets where the data will be located.

You can know navigate back to the Amazon SageMaker console, then to the Notebook Instances menu. Start your instance and launch either Jupyter or JupyterLab session. From there, you can launch a new terminal and clone this repository into your local development machine using `git clone`.

### Repository structure
Once you've cloned this repo, browse to the [**data preparation**](notebooks/1_data_preparation.ipynb) notebook: this first notebook will download and prepare the data necessary for the other ones.

```
.
|
+-- README.md                          <-- This instruction file
|
+-- assets/                            <-- Pictures used throughout the notebooks
|
+-- dataset/                           <-- Raw dataset files are located here
|   |-- labels.csv                         | Known anomaly ranges
|   |-- tags_description.csv               | Description of each time serie signal
|   |-- timeranges.txt                     | Train / evaluation split
|   \-- timeseries.zip                     | Raw timeseries
|
+-- notebooks/
|   |-- 1_data_preparation.ipynb       <-- START HERE: data preparation notebook, useful to
|   |                                      download and prepare the data, get familiar with
|   |                                      them
|   |
|   |-- 2_dataset_creation.ipynb       <-- Create a Lookout for Equipment dataset
|   |-- 3_model_training.ipynb         <-- Train a Lookout for Equipment model
|   |-- 4_model_evaluation.ipynb       <-- Plot the evaluation results and some diagnostics
|   \-- 5_inference_scheduling.ipynb   <-- Schedule a regular inference execution
|
+-- utils/
    |-- lookout_equipment_utils.py     <-- Utilities to manage Lookout for Equipment assets
    \-- lookoutequipment.json          <-- Configuration file to access the API while the service
                                           is in preview
```

## Questions

Please contact [**Michaël HOARAU**](mailto:michoara@amazon.fr) or raise an issue on this repository.

## Security

See [**CONTRIBUTING**](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
This collection of notebooks is licensed under the MIT-0 License. See the LICENSE file.