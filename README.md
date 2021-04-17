# Amazon Lookout For Equipment
---

## Overview

Amazon Lookout for Equipment uses the data from your sensors to detect abnormal equipment behavior, so you can take action before machine failures occur and avoid unplanned downtime.

Amazon Lookout for Equipment analyzes data such as pressure, flow rate, RPMs, temperature, and power to automatically train a specific ML model based on just your data, for your equipment – with no ML expertise required.

Lookout for Equipment uses your unique ML model to analyze incoming sensor data in real-time and accurately identify early warning signs that could lead to machine failures.

This means you can detect equipment abnormalities with speed and precision, quickly diagnose issues, take action to reduce expensive downtime, and reduce false alerts.

## Demo notebooks
This folder contains various examples covering Amazon Lookout for Equipment best
practices. Open the **[getting_started](getting_started)** folder to find all the
ressources you need to train your first anomaly detection model. The notebooks 
provided can also serve as a template to build your own models with your own data.

In the **[getting_started](getting_started)** folder, you will learn to:

1. Prepare your data for use with Amazon Lookout for Equipment
2. Create your own dataset
3. Train a model based on this dataset
4. Evaluate a model performance and get some diagnostics based on historical data
5. Build an inference scheduler and post-process the predictions

## Blogs

In this folder, you will find technical content associated to blog posts
AWS writes about Amazon Lookout for Equipment.

## Preprocessing

Multivariate industrial time series data can be challenging to deal
with: these samples will show you how to explore your data, improve
data quality, label your anomalies (manually or automatically), etc.

## Security

See [**CONTRIBUTING**](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This collection of notebooks is licensed under the MIT-0 License. See the
[**LICENSE**](LICENSE) file.