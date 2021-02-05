# Standard python and AWS imports:
import boto3
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
import time
import uuid

from botocore.config import Config
from datetime import datetime
from matplotlib.dates import DateFormatter
from matplotlib import gridspec
from scipy.stats import wasserstein_distance
from typing import List, Dict
from tqdm import tqdm

# Parameters
DEFAULT_REGION = 'eu-west-1'

def get_client(region_name=DEFAULT_REGION):
    """
    Get a boto3 client for the Amazon Lookout for Equipment service.
    
    PARAMS
    ======
        region_name: string
            AWS region name. (Default: eu-west-1)
    
    RETURN
    ======
        lookoutequipment_client
            A boto3 client to interact with the L4E service
    """
    lookoutequipment_client = boto3.client(
        service_name='lookoutequipment',
        region_name=region_name,
        config=Config(
            connect_timeout=30, 
            read_timeout=30, 
            retries={'max_attempts': 3}
        ),
        endpoint_url=f'https://lookoutequipment.{region_name}.amazonaws.com/'
    )
    
    return lookoutequipment_client


def list_datasets(
    dataset_name_prefix=None,
    max_results=50,
    region_name=DEFAULT_REGION
):
    """
    List all the Lookout for Equipment datasets available in this account.
    
    PARAMS
    ======
        dataset_name_prefix: string
            Prefix to filter out all the datasets which names starts by 
            this prefix. Defaults to None to list all datasets.
            
        max_results: integer (default: 50)
            Max number of datasets to return 
            
        region_name: string
            AWS region name. (Default: eu-west-1)
            
    RETURN
    ======
        dataset_list: list of strings
            A list with all the dataset names found in the current region
    """
    # Initialization:
    dataset_list = []
    has_more_records = True
    lookoutequipment_client = get_client(region_name=region_name)
    
    # Building the request:
    kargs = {"MaxResults": max_results}
    if dataset_name_prefix is not None:
        kargs["DatasetNameBeginsWith"] = dataset_name_prefix
    
    # We query for the list of datasets, until there are none left to fetch:
    while has_more_records:
        # Query for the list of L4E datasets available for this AWS account:
        list_datasets_response = lookoutequipment_client.list_datasets(**kargs)
        if "NextToken" in list_datasets_response:
            kargs["NextToken"] = list_datasets_response["NextToken"]
        else:
            has_more_records = False
        
        # Add the dataset names to the list:
        dataset_summaries = list_datasets_response["DatasetSummaries"]
        for dataset_summary in dataset_summaries:
            dataset_list.append(dataset_summary['DatasetName'])
    
    return dataset_list

def list_models_for_datasets(
    model_name_prefix=None, 
    dataset_name_prefix=None,
    max_results=50,
    region_name=DEFAULT_REGION
):
    """
    List all the models available in a given region.
    
    PARAMS
    ======
        model_name_prefix: string (default: None)
            Prefix to filter on the model name to look for
            
        dataset_name_prefix: string (default None)
            Prefix to filter the dataset name: if used, only models
            making use of this particular dataset are returned

        max_results: integer (default: 50)
            Max number of datasets to return 
            
    RETURNS
    =======
        models_list: list of string
            List of all the models corresponding to the input parameters
            (regions and dataset)
    """
    # Initialization:
    models_list = []
    has_more_records = True
    client = get_client(region_name=region_name)
    
    # Building the request:
    list_models_request = {"MaxResults": max_results}
    if model_name_prefix is not None:
        list_models_request["ModelNameBeginsWith"] = model_name_prefix
    if dataset_name_prefix is not None:
        list_models_request["DatasetNameBeginsWith"] = dataset_name_prefix

    # We query for the list of models, until there are none left to fetch:
    while has_more_records:
        # Query for the list of L4E models available for this AWS account:
        list_models_response = client.list_models(**list_models_request)
        if "NextToken" in list_models_response:
            list_models_request["NextToken"] = list_models_response["NextToken"]
        else:
            has_more_records = False

        # Add the model names to the list:
        model_summaries = list_models_response["ModelSummaries"]
        for model_summary in model_summaries:
            models_list.append(model_summary['ModelName'])

    return models_list

def create_data_schema(component_fields_map: Dict):
    """
    Generates a JSON formatted string from a dictionary
    
    PARAMS
    ======
        component_fields_map: dict
            A dictionary containing a field maps for the dataset schema
            
    RETURNS
    =======
        schema: string
            A JSON-formatted string ready to be used as a schema for a dataset
    """
    schema = json.dumps(
        _create_data_schema_map(
            component_fields_map=component_fields_map
        )
    )
    
    return schema

def _create_data_schema_map(component_fields_map: Dict):
    """
    Generate a dictionary with the JSON format expected by Lookout for Equipment
    to be used as the schema for a dataset at ingestion, training time and
    inference time
    
    PARAMS
    ======
        component_fields_map: dict
            A dictionary containing a field maps for the dataset schema

    RETURNS
    =======
        data_schema: dict
            A dictionnary containing the detailed schema built from the original
            dictionary mapping
    """
    # The root of the schema is a "Components" tag:
    data_schema = dict()
    component_schema_list = list()
    data_schema['Components'] = component_schema_list

    # We then loop through each first level tag from the dictionary:
    for component_name in component_fields_map:
        # We create a schema for the current component:
        component_schema = _create_component_schema(
            component_name, 
            component_fields_map[component_name]
        )
        
        # And update the overall dictionary:
        component_schema_list.append(component_schema)

    return data_schema

def _create_component_schema(component_name: str, field_names: List):
    """
    Build a schema for a given component and fieds list
    
    PARAMS
    ======
        component_name: string
            Name of the component to build a schema for
        
        field_names: list of strings
            Name of all the fields included in this component
            
    RETURNS
    =======
        component_schema: dict 
            A dictionnary containing the detailed schema for a given component
    """
    # Initialize the dictionary with the component name:
    component_schema = dict()
    component_schema['ComponentName'] = component_name
    
    # Loops through all the fields:
    col_list = []
    is_first_field = True
    component_schema['Columns'] = col_list
    for field_name in field_names:
        # The first field is considered to be the timestamp:
        if is_first_field:
            ts_col = dict()
            ts_col['Name'] = field_name
            ts_col['Type'] = 'DATETIME'
            col_list.append(ts_col)
            is_first_field = False
            
        # All the other fields are supposed to be float values:
        else:
            attr_col = dict()
            attr_col['Name'] = field_name
            attr_col['Type'] = 'DOUBLE'
            col_list.append(attr_col)
            
    return component_schema
    
def plot_timeseries(
    timeseries_df,
    tag_name, 
    start=None,
    end=None, 
    plot_rolling_avg=False, 
    labels_df=None, 
    predictions=None,
    tag_split=None,
    custom_grid=True,
    fig_width=18,
    prediction_titles=None
):
    """
    This function plots a time series signal with a line plot and can combine
    this with labelled and predicted anomaly ranges.
    
    PARAMS
    ======
        timeseries_df: pandas.DataFrame
            A dataframe containing the time series to plot
        
        tag_name: string
            The name of the tag that we can add in the label
        
        start: string or pandas.Datetime (default: None)
            Starting timestamp of the signal to plot. If not provided, will use
            the whole signal
        
        end: string or pandas.Datetime (default: None)
            End timestamp of the signal to plot. If not provided, will use the
            whole signal
        
        plot_rolling_avg: boolean (default: False)
            If set to true, will add a rolling average curve on top of the
            line plot for the time series.
        
        labels_df: pandas.DataFrame (default: None)
            If provided, this is a dataframe with all the labelled anomalies.
            This will be rendered as a filled-in plots below the time series
            itself.
        
        predictions: pandas.DataFrame or list of pandas.DataFrame
            If provided, this is a dataframe with all the predicted anomalies.
            This will be rendered as a filled-in plots below the time series
            itself.
            
        tag_split: string or pandas.Datetime
            If provided, the line plot will plot the first part of the time
            series with a colour and the second part in grey. This can be
            used to show the split between training and evaluation period for
            instance.
        
        custom_grid: boolean (default: True)
            Will show a custom grid with month name mentionned for each quarter
            and lighter lines for the other month to prevent clutter on the
            horizontal axis.
        
        fig_width: integer (default: 18)
            Figure width.
        
        prediction_titles: list of strings (default: None)
            If we want to plot multiple predictions, we can set the titles for
            each of the prediction plot.
    
    RETURNS
    =======
        fig: matplotlib.pyplot.figure
            A figure where the plots are drawn
            
        ax: matplotlib.pyplot.Axis
            An axis where the plots are drawn
    """
    if start is None:
        start = timeseries_df.index.min()
    elif type(start) == str:
        start = pd.to_datetime(start)
        
    if end is None:
        end = timeseries_df.index.max()
    elif type(end) == str:
        end = pd.to_datetime(end)
        
    if (tag_split is not None) & (type(tag_split) == str):
        tag_split = pd.to_datetime(tag_split)

    # Prepare the figure:
    fig_height = 4
    height_ratios = [8]
    nb_plots = 1
    
    if labels_df is not None:
        fig_height += 1
        height_ratios += [1.5]
        nb_plots += 1
        
    if predictions is not None:
        if type(predictions) == pd.core.frame.DataFrame:
            fig_height += 1
            height_ratios += [1.5]
            nb_plots += 1
        elif type(predictions) == list:
            fig_height += 1 * len(predictions)
            height_ratios = height_ratios + [1.5] * len(predictions)
            nb_plots += len(predictions)
            
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nb_plots, 1, height_ratios=height_ratios, hspace=0.5)
    ax = []
    for i in range(nb_plots):
        ax.append(fig.add_subplot(gs[i]))
        
    # Plot the time series signal:
    data = timeseries_df[start:end].copy()
    if tag_split is not None:
        ax[0].plot(data.loc[start:tag_split, 'Value'], 
                   linewidth=0.5, 
                   alpha=0.5, 
                   label=f'{tag_name} - Training', 
                   color='tab:grey')
        ax[0].plot(data.loc[tag_split:end, 'Value'], 
                   linewidth=0.5, 
                   alpha=0.8, 
                   label=f'{tag_name} - Evaluation')
    else:
        ax[0].plot(data['Value'], linewidth=0.5, alpha=0.8, label=tag_name)
    ax[0].set_xlim(start, end)
    
    # Plot a daily rolling average:
    if plot_rolling_avg == True:
        daily_rolling_average = data['Value'].rolling(window=60*24).mean()
        ax[0].plot(data.index, 
                   daily_rolling_average, 
                   alpha=0.5, 
                   color='white', 
                   linewidth=3)
        ax[0].plot(data.index, 
                   daily_rolling_average, 
                   label='Daily rolling leverage', 
                   color='tab:red', 
                   linewidth=1)

    # Configure custom grid:
    ax_id = 0
    if custom_grid:
        date_format = DateFormatter("%Y-%m")
        major_ticks = np.arange(start, end, 3, dtype='datetime64[M]')
        minor_ticks = np.arange(start, end, 1, dtype='datetime64[M]')
        ax[ax_id].xaxis.set_major_formatter(date_format)
        ax[ax_id].set_xticks(major_ticks)
        ax[ax_id].set_xticks(minor_ticks, minor=True)
        ax[ax_id].grid(which='minor', axis='x', alpha=0.8)
        ax[ax_id].grid(which='major', axis='x', alpha=1.0, linewidth=2)
        ax[ax_id].xaxis.set_tick_params(rotation=30)

    # Add the labels on a second plot:
    if labels_df is not None:
        ax_id += 1
        label_index = pd.date_range(
            start=data.index.min(), 
            end=data.index.max(), 
            freq='1min'
        )
        label_data = pd.DataFrame(index=label_index)
        label_data.loc[:, 'Label'] = 0.0

        for index, row in labels_df.iterrows():
            event_start = row['start']
            event_end = row['end']
            label_data.loc[event_start:event_end, 'Label'] = 1.0
            
        ax[ax_id].plot(label_data['Label'], color='tab:green', linewidth=0.5)
        ax[ax_id].set_xlim(start, end)
        ax[ax_id].fill_between(label_index, 
                               y1=label_data['Label'], 
                               y2=0, 
                               alpha=0.1, 
                               color='tab:green', 
                               label='Real anomaly range (label)')
        ax[ax_id].axes.get_xaxis().set_ticks([])
        ax[ax_id].axes.get_yaxis().set_ticks([])
        ax[ax_id].set_xlabel('Anomaly ranges (labels)', fontsize=12)
        
    # Add the labels (anomaly range) on a 
    # third plot located below the main ones:
    if predictions is not None:
        pred_index = pd.date_range(
            start=data.index.min(), 
            end=data.index.max(), 
            freq='1min')
        pred_data = pd.DataFrame(index=pred_index)
        
        if type(predictions) == pd.core.frame.DataFrame:
            ax_id += 1
            pred_data.loc[:, 'prediction'] = 0.0

            for index, row in predictions.iterrows():
                event_start = row['start']
                event_end = row['end']
                pred_data.loc[event_start:event_end, 'prediction'] = 1.0

            ax[ax_id].plot(pred_data['prediction'], 
                           color='tab:red',
                           linewidth=0.5)
            ax[ax_id].set_xlim(start, end)
            ax[ax_id].fill_between(pred_index, 
                                   y1=pred_data['prediction'],
                                   y2=0, 
                                   alpha=0.1, 
                                   color='tab:red')
            ax[ax_id].axes.get_xaxis().set_ticks([])
            ax[ax_id].axes.get_yaxis().set_ticks([])
            ax[ax_id].set_xlabel('Anomaly ranges (Prediction)', fontsize=12)
            
        elif type(predictions) == list:
            for prediction_index, p in enumerate(predictions):
                ax_id += 1
                pred_data.loc[:, 'prediction'] = 0.0

                for index, row in p.iterrows():
                    event_start = row['start']
                    event_end = row['end']
                    pred_data.loc[event_start:event_end, 'prediction'] = 1.0
                
                ax[ax_id].plot(pred_data['prediction'], 
                               color='tab:red',
                               linewidth=0.5)
                ax[ax_id].set_xlim(start, end)
                ax[ax_id].fill_between(pred_index,
                                       y1=pred_data['prediction'],
                                       y2=0, 
                                       alpha=0.1, 
                                       color='tab:red')
                ax[ax_id].axes.get_xaxis().set_ticks([])
                ax[ax_id].axes.get_yaxis().set_ticks([])
                ax[ax_id].set_xlabel(
                    prediction_titles[prediction_index], 
                    fontsize=12
                )
        
    # Show the plot with a legend:
    ax[0].legend(fontsize=10, loc='upper right', framealpha=0.4)
        
    return fig, ax

class LookoutEquipmentAnalysis:
    """
    A class to manage Lookout for Equipment result analysis
    
    ATTRIBUTES
    ==========
        model_name: string
            The name of the Lookout for Equipment trained model
                
        predicted_ranges: pandas.DataFrame
            A Pandas dataframe with the predicted anomaly ranges listed in
            chronological order with a Start and End columns

        labelled_ranges: pandas.DataFrame
            A Pandas dataframe with the labelled anomaly ranges listed in
            chronological order with a Start and End columns

        df_list: list of pandas.DataFrame
            A list with each time series into a dataframe

    METHODS
    =======
        set_time_periods():
            Sets the time period used for the analysis of the model evaluations
            
        get_predictions():
            Get the anomaly ranges predicted by the current model
            
        get_labels():
            Get the labelled ranges as provided to the model before training
            
        compute_histograms():
            This method loops through each signal and computes two distributions
            of the values in the time series: one for all the anomalies found in
            the evaluation period and another one with all the normal values 
            found in the same period. It then ranks every signals based on the
            distance between these two histograms

        plot_histograms():
            Plot the top 12 signal values distribution by decreasing ranking 
            distance (as computed by the compute_histograms() method
            
        plot_signals():
            Plot the top 12 signals by decreasing ranking distance. For each 
            signal, this method will plot the normal values in green and the 
            anomalies in red

        get_ranked_list():
            Returns the list of signals with computed rank
    """
    def __init__(self, model_name, tags_df, region_name=DEFAULT_REGION):
        """
        Create a new analysis for a Lookout for Equipment model.
        
        PARAMS
        ======
            model_name: string
                The name of the Lookout for Equipment trained model
                
            tags_df: pandas.DataFrame
                A dataframe containing all the signals, indexed by time
                
            region_name: string
                Name of the AWS region from where the service is called.
        """
        self.client = get_client(region_name)
        self.model_name = model_name
        self.predicted_ranges = None
        self.labelled_ranges = None
        
        self.ts_normal_training = None
        self.ts_label_evaluation = None
        self.ts_known_anomalies = None
        
        self.df_list = dict()
        for signal in tags_df.columns:
            self.df_list.update({signal: tags_df[[signal]]})
            
        # Extracting time ranges used at training time:
        model_description = self.client.describe_model(ModelName=self.model_name)
        self.training_start = pd.to_datetime(
            model_description['TrainingDataStartTime'].replace(tzinfo=None)
        )
        self.training_end = pd.to_datetime(
            model_description['TrainingDataEndTime'].replace(tzinfo=None)
        )
        self.evaluation_start = pd.to_datetime(
            model_description['EvaluationDataStartTime'].replace(tzinfo=None)
        )
        self.evaluation_end = pd.to_datetime(
            model_description['EvaluationDataEndTime'].replace(tzinfo=None)
        )

    def _load_model_response(self):
        """
        Use the trained model description to extract labelled and predicted 
        ranges of anomalies. This method will extract them from the 
        DescribeModel API from Lookout for Equipment and store them in the
        labelled_ranges and predicted_ranges properties.
        """
        describe_model_response = self.client.describe_model(
            ModelName=self.model_name
        )
        
        if self.labelled_ranges is None:
            self.labelled_ranges = eval(
                describe_model_response['ModelMetrics']
            )['labeled_ranges']
            self.labelled_ranges = pd.DataFrame(self.labelled_ranges)
            self.labelled_ranges['start'] = pd.to_datetime(self.labelled_ranges['start'])
            self.labelled_ranges['end'] = pd.to_datetime(self.labelled_ranges['end'])
            
        self.predicted_ranges = eval(
            describe_model_response['ModelMetrics']
        )['predicted_ranges']
        self.predicted_ranges = pd.DataFrame(self.predicted_ranges)
        self.predicted_ranges['start'] = pd.to_datetime(self.predicted_ranges['start'])
        self.predicted_ranges['end'] = pd.to_datetime(self.predicted_ranges['end'])
        
    def set_time_periods(
        self, 
        evaluation_start, 
        evaluation_end, 
        training_start, 
        training_end
    ):
        """
        Set the time period of analysis
        
        PARAMS
        ======
            evaluation_start: datetime
                Start of the evaluation period

            evaluation_end: datetime
                End of the evaluation period

            training_start: datetime
                Start of the training period

            training_end: datetime
                End of the training period
        """
        self.evaluation_start = evaluation_start
        self.evaluation_end = evaluation_end
        self.training_start = training_start
        self.training_end = training_end
    
    def get_predictions(self):
        """
        Get the anomaly ranges predicted by the current model
        
        RETURN
        ======
            predicted_ranges: pandas.DataFrame
                A Pandas dataframe with the predicted anomaly ranges listed in
                chronological order with a Start and End columns
        """
        if self.predicted_ranges is None:
            self._load_model_response()
            
        return self.predicted_ranges
        
    def get_labels(self, labels_fname=None):
        """
        Get the labelled ranges as provided to the model before training
        
        PARAMS
        ======
            labels_fname: string (Default to None)
                As an option, if you provide a path to a CSV file containing
                the label ranges, this method will use this file to load the
                labels. If this argument is not provided, it will load the
                labels from the trained model Describe API
        
        RETURN
        ======
            labelled_ranges: pandas.DataFrame
                A Pandas dataframe with the labelled anomaly ranges listed in
                chronological order with a Start and End columns
        """
        if labels_fname is not None:
            labels_df = pd.read_csv(labels_fname, header=None)
            labels_df[0] = pd.to_datetime(labels_df[0])
            labels_df[1] = pd.to_datetime(labels_df[1])
            labels_df.columns = ['start', 'end']
            self.labelled_ranges = labels_df
        
        elif self.labelled_ranges is None:
            self._load_model_response()
            
        return self.labelled_ranges
    
    def _get_time_ranges(self):
        """
        Extract DateTimeIndex with normal values and anomalies from the
        predictions generated by the model.
        
        RETURNS
        =======
            index_normal: pandas.DateTimeIndex
                Timestamp index for all the values marked as normal during the
                training period
                
            index_anomaly: pandas.DateTimeIndex
                Timestamp index for all the values predicted as anomalies by
                the model during the evaluation period
        """
        # Extract the first time series 
        tag = list(self.df_list.keys())[0]
        tag_df = self.df_list[tag]
        
        # Initialize the predictions dataframe:
        predictions_df = pd.DataFrame(columns=['Prediction'], index=tag_df.index)
        predictions_df['Prediction'] = 0

        # Loops through the predicted and labelled anomalies
        # ranges and set these predictions to 1 (predicted) 
        # or 2 (initially known):
        for index, row in self.predicted_ranges.iterrows():
            predictions_df.loc[row['start']:row['end'], 'Prediction'] = 1
        for index, row in self.labelled_ranges.iterrows():
            predictions_df.loc[row['start']:row['end'], 'Prediction'] = 2

        # Limits the analysis range to the evaluation period:
        predictions_df = predictions_df[self.training_start:self.evaluation_end]
        
        # Build a DateTimeIndex for normal values and anomalies:
        index_normal = predictions_df[predictions_df['Prediction'] == 0].index
        index_anomaly = predictions_df[predictions_df['Prediction'] == 1].index
        index_known = predictions_df[predictions_df['Prediction'] == 2].index
        
        return index_normal, index_anomaly, index_known
    
    def compute_histograms(
        self, 
        index_normal=None, 
        index_anomaly=None, 
        num_bins=20
    ):
        """
        This method loops through each signal and computes two distributions of
        the values in the time series: one for all the anomalies found in the
        evaluation period and another one with all the normal values found in the
        same period. It then computes the Wasserstein distance between these two
        histograms and then rank every signals based on this distance. The higher
        the distance, the more different a signal is when comparing anomalous
        and normal periods. This can orient the investigation of a subject 
        matter expert towards the sensors and associated components.
        
        PARAMS
        ======
            index_normal: pandas.DateTimeIndex
                All the normal indices
                
            index_anomaly: pandas.DateTimeIndex
                All the indices for anomalies
                
            num_bins: integer (default: 20)
                Number of bins to use to build the distributions
        """
        if (index_normal is None) or (index_anomaly is None):
            index_lists = self._get_time_ranges()
            self.ts_normal_training = index_lists[0]
            self.ts_label_evaluation = index_lists[1]
            self.ts_known_anomalies = index_lists[2]

        self.num_bins = num_bins

        # Now we loop on each signal to compute a 
        # histogram of each of them in this anomaly range,
        # compte another one in the normal range and
        # compute a distance between these:
        rank = dict()
        for tag, current_tag_df in tqdm(
            self.df_list.items(), 
            desc='Computing distributions'
        ):
            try:
                # Get the values for the whole signal, parts
                # marked as anomalies and normal part:
                current_signal_values = current_tag_df[tag]
                current_signal_evaluation = current_tag_df.loc[self.ts_label_evaluation, tag]
                current_signal_training = current_tag_df.loc[self.ts_normal_training, tag]

                # Let's compute a bin width based on the whole range of possible 
                # values for this signal (across the normal and anomalous periods).
                # For both normalization and aesthetic reasons, we want the same
                # number of bins across all signals:
                bin_width = (np.max(current_signal_values) - np.min(current_signal_values))/self.num_bins
                bins = np.arange(
                    np.min(current_signal_values), 
                    np.max(current_signal_values) + bin_width, 
                    bin_width
                )

                # We now use the same bins arrangement for both parts of the signal:
                u = np.histogram(
                    current_signal_training, 
                    bins=bins, 
                    density=True
                )[0]
                v = np.histogram(
                    current_signal_evaluation, 
                    bins=bins, 
                    density=True
                )[0]

                # Wasserstein distance is the earth mover distance: it can be 
                # used to compute a similarity between two distributions: this
                # metric is only valid when the histograms are normalized (hence
                # the density=True in the computation above):
                d = wasserstein_distance(u, v)
                rank.update({tag: d})

            except Exception as e:
                rank.update({tag: 0.0})

        # Sort histograms by decreasing Wasserstein distance:
        rank = {k: v for k, v in sorted(rank.items(), key=lambda rank: rank[1], reverse=True)}
        self.rank = rank
        
    def plot_histograms(self, nb_cols=3, max_plots=12):
        """
        Once the histograms are computed, we can plot the top N by decreasing 
        ranking distance. By default, this will plot the histograms for the top
        12 signals, with 3 plots per line.
        
        PARAMS
        ======
            nb_cols: integer (default: 3)
                Number of plots to assemble on a given row
                
            max_plots: integer (default: 12)
                Number of signal to consider
                
        RETURNS
        =======
            fig: matplotlib.pyplot.figure
                A figure where the plots are drawn
                
            axes: list of matplotlib.pyplot.Axis
                An axis for each plot drawn here
        """
        # Prepare the figure:
        nb_rows = len(self.df_list.keys()) // nb_cols + 1
        plt.style.use('Solarize_Light2')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fig = plt.figure(figsize=(16, int(nb_rows * 3)))
        gs = gridspec.GridSpec(nb_rows, nb_cols, hspace=0.5, wspace=0.25)
        axes = []
        for i in range(max_plots):
            axes.append(fig.add_subplot(gs[i]))

        # Loops through each signal by decreasing distance order:
        i = 0
        for tag, current_rank in tqdm(
            self.rank.items(), 
            total=max_plots, 
            desc='Preparing histograms'
        ):
            # We stop after reaching the number of plots we are interested in:
            if i > max_plots - 1:
                break

            try:
                # Get the anomaly and the normal values from the current signal:
                current_signal_values = self.df_list[tag][tag]
                current_signal_evaluation = self.df_list[tag].loc[self.ts_label_evaluation, tag]
                current_signal_training = self.df_list[tag].loc[self.ts_normal_training, tag]

                # Compute the bin width and bin edges to match the 
                # number of bins we want to have on each histogram:
                bin_width =(np.max(current_signal_values) - np.min(current_signal_values))/self.num_bins
                bins = np.arange(
                    np.min(current_signal_values), 
                    np.max(current_signal_values) + bin_width, 
                    bin_width
                )

                # Add both histograms in the same plot:
                axes[i].hist(current_signal_training, 
                         density=True, 
                         alpha=0.5, 
                         color=colors[1], 
                         bins=bins, 
                         edgecolor='#FFFFFF')
                axes[i].hist(current_signal_evaluation, 
                         alpha=0.5, 
                         density=True, 
                         color=colors[5], 
                         bins=bins, 
                         edgecolor='#FFFFFF')

            except Exception as e:
                print(e)
                axes[i] = plt.subplot(gs[i])

            # Removes all the decoration to leave only the histograms:
            axes[i].grid(False)
            axes[i].get_yaxis().set_visible(False)
            axes[i].get_xaxis().set_visible(False)

            # Title will be the tag name followed by the score:
            title = tag
            title += f' (score: {current_rank:.02f})'
            axes[i].set_title(title, fontsize=10)

            i+= 1
            
        return fig, axes
            
    def plot_signals(self, nb_cols=3, max_plots=12):
        """
        Once the histograms are computed, we can plot the top N signals by 
        decreasing ranking distance. By default, this will plot the signals for 
        the top 12 signals, with 3 plots per line. For each signal, this method
        will plot the normal values in green and the anomalies in red.
        
        PARAMS
        ======
            nb_cols: integer (default: 3)
                Number of plots to assemble on a given row
                
            max_plots: integer (default: 12)
                Number of signal to consider
                
        RETURNS
        =======
            fig: matplotlib.pyplot.figure
                A figure where the plots are drawn
                
            axes: list of matplotlib.pyplot.Axis
                An axis for each plot drawn here
        """
        # Prepare the figure:
        nb_rows = max_plots // nb_cols + 1
        plt.style.use('Solarize_Light2')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fig = plt.figure(figsize=(28, int(nb_rows * 4)))
        gs = gridspec.GridSpec(nb_rows, nb_cols, hspace=0.5, wspace=0.25)
        axes = []
        for i in range(max_plots):
            axes.append(fig.add_subplot(gs[i]))
        
        # Loops through each signal by decreasing distance order:
        i = 0
        for tag, current_rank in self.rank.items():
            # We stop after reaching the number of plots we are interested in:
            if i > max_plots - 1:
                break

            # Get the anomaly and the normal values from the current signal:
            current_signal_evaluation = self.df_list[tag].loc[self.ts_label_evaluation, tag]
            current_signal_training = self.df_list[tag].loc[self.ts_normal_training, tag]
            current_signal_known = self.df_list[tag].loc[self.ts_known_anomalies, tag]

            # Plot both time series with a line plot
            # axes.append(plt.subplot(gs[i]))
            axes[i].plot(current_signal_training, 
                         linewidth=0.5, 
                         alpha=0.8, 
                         color=colors[1])
            axes[i].plot(current_signal_evaluation, 
                         linewidth=0.5, 
                         alpha=0.8, 
                         color=colors[5])
            axes[i].plot(current_signal_known, 
                         linewidth=0.5, 
                         alpha=0.8, 
                         color='#AAAAAA')

            # Title will be the tag name followed by the score:
            title = tag
            title += f' (score: {current_rank:.02f})'
            axes[i].set_title(title, fontsize=10)
            start = min(
                min(self.ts_label_evaluation),
                min(self.ts_normal_training), 
                min(self.ts_known_anomalies)
            )
            end = max(
                max(self.ts_label_evaluation),
                max(self.ts_normal_training), 
                max(self.ts_known_anomalies)
            )
            axes[i].set_xlim(start, end)

            i += 1
            
        return fig, axes
            
    def get_ranked_list(self, max_signals=12):
        """
        Returns the list of signals with computed rank.
        
        PARAMS
        ======
            max_signals: integer (default: 12)
                Number of signals to consider
        
        RETURNS
        =======
            significant_signals_df: pandas.DataFrame
                A dataframe with each signal and the associated rank value
        """
        significant_signals_df = pd.DataFrame(list(self.rank.items())[:max_signals])
        significant_signals_df.columns = ['Tag', 'Rank']
        
        return significant_signals_df
    
class LookoutEquipmentScheduler:
    """
    A class to represent a Lookout for Equipment inference scheduler object.
    
    ATTRIBUTES
    ==========
        scheduler_name: string
            Name of the scheduler associated to this object
            
        model_name: string
            Name of the model used to run the inference when the scheduler
            wakes up
            
        create_request: dict
            A dictionary containing all the parameters to configure and create
            an inference scheduler
            
        execution_summaries: list of dict
            A list of all inference execution results. Each execution is stored
            as a dictionary.

    METHODS
    =======
        set_parameters():
            Sets all the parameters necessary to manage this scheduler object
        
        create():
            Creates a new scheduler
            
        start():
            Start an existing scheduler
            
        stop():
            Stop an existing scheduler
            
        delete():
            Detele a stopped scheduler
            
        get_status():
            Returns the status of the scheduler
            
        list_inference_executions():
            Returns all the results from the inference executed by the scheduler
            
        get_predictions():
            Return the predictions generated by the executed inference
    """
    def __init__(self, scheduler_name, model_name, region_name=DEFAULT_REGION):
        """
        Constructs all the necessary attributes for a scheduler object.
        
        PARAMS
        ======
            scheduler_name: string
                The name of the scheduler to be created or managed
                
            model_name: string
                The name of the model to schedule inference for
                
            region_name: string
                Name of the AWS region from where the service is called.
        """
        self.scheduler_name = scheduler_name
        self.model_name = model_name
        self.client = get_client(region_name)
        
        self.create_request = dict()
        self.create_request.update({'ModelName': model_name})
        self.create_request.update({'InferenceSchedulerName': scheduler_name})

        self.execution_summaries = None
        
    def set_parameters(self,
                       input_bucket,
                       input_prefix,
                       output_bucket,
                       output_prefix,
                       role_arn,
                       upload_frequency='PT5M',
                       delay_offset=None,
                       timezone_offset='+00:00',
                       component_delimiter='_',
                       timestamp_format='yyyyMMddHHmmss'
                      ):
        """
        Set all the attributes for the scheduler object.
        
        PARAMS
        ======
            input_bucket: string
                Bucket when the input data are located

            input_prefix: string
                Location prefix for the input data
                
            output_bucket: string
                Bucket location for the inference execution output
                
            output_prefix: string
                Location prefix for the inference result file
                
            role_arn: string
                Role allowing Lookout for Equipment to read and write data
                from the input and output bucket locations
                
            upload_frequency: string (default: PT5M)
                Upload frequency of the data
            
            delay_offset: integer (default: None)
                Offset in minute, ensuring the data are available when the
                scheduler wakes up to run the inference
            
            timezone_offset: string (default: +00:00)
                Timezone offset used to match the date in the input filenames
            
            component_delimiter: string (default: '_')
                Character to use to delimit component name and timestamp in the
                input filenames
            
            timestamp_format: string (default: yyyyMMddHHmmss)
                Format of the timestamp to look for in the input filenames
        """
        # Configure the mandatory parameters:
        self.create_request.update({'DataUploadFrequency': upload_frequency})
        self.create_request.update({'RoleArn': role_arn})
        
        # Configure the optional parameters:
        if delay_offset is not None:
            self.create_request.update({'DataDelayOffsetInMinutes': delay_offset})
            
        # Setup data input configuration:
        inference_input_config = dict()
        inference_input_config['S3InputConfiguration'] = dict([
            ('Bucket', input_bucket),
            ('Prefix', input_prefix)
        ])
        if timezone_offset is not None:
            inference_input_config['InputTimeZoneOffset'] = timezone_offset
        if component_delimiter is not None or timestamp_format is not None:
            input_name_cfg = dict()
            if component_delimiter is not None:
                input_name_cfg['ComponentTimestampDelimiter'] = component_delimiter
            if timestamp_format is not None:
                input_name_cfg['TimestampFormat'] = timestamp_format
            inference_input_config['InferenceInputNameConfiguration'] = input_name_cfg
        self.create_request.update({
            'DataInputConfiguration': inference_input_config
        })

        #  Set up output configuration:
        inference_output_config = dict()
        inference_output_config['S3OutputConfiguration'] = dict([
            ('Bucket', output_bucket),
            ('Prefix', output_prefix)
        ])
        self.create_request.update({
            'DataOutputConfiguration': inference_output_config
        })
        
    def _poll_event(self, scheduler_status, wait_state, sleep_time=5):
        """
        Wait for a given scheduler update process to be finished
        
        PARAMS
        ======
            scheduler_status: string
                Initial scheduler status
            
            wait_state: string (either PENDING, STOPPING)
                The wait will continue while the status has a value equal
                to this wait_state string
                
            sleep_time: integer (default: 5)
                How many seconds should we wait before polling again
        """
        print("===== Polling Inference Scheduler Status =====\n")
        print("Scheduler Status: " + scheduler_status)
        while scheduler_status == wait_state:
            time.sleep(sleep_time)
            describe_scheduler_response = self.client.describe_inference_scheduler(
                InferenceSchedulerName=self.scheduler_name
            )
            scheduler_status = describe_scheduler_response['Status']
            print("Scheduler Status: " + scheduler_status)
        print("\n===== End of Polling Inference Scheduler Status =====")

    def create(self, wait=True):
        """
        Create an inference scheduler for a trained Lookout for Equipment model
        """
        # Update the creation request:
        self.create_request.update({'ClientToken': uuid.uuid4().hex})
        
        # Create the scheduler:
        create_scheduler_response = self.client.create_inference_scheduler(
            **self.create_request
        )
        
        # Polling scheduler creation status:
        if wait:
            self._poll_event(create_scheduler_response['Status'], 'PENDING')
    
    def start(self, wait=True):
        """
        Start an existing inference scheduler if it exists
        """
        start_scheduler_response = self.client.start_inference_scheduler(
            InferenceSchedulerName=self.scheduler_name
        )

        # Wait until started:
        if wait:
            self._poll_event(start_scheduler_response['Status'], 'PENDING')
        
    def stop(self, wait=True):
        """
        Stop an existing started inference scheduler
        """
        stop_scheduler_response = self.client.stop_inference_scheduler(
            InferenceSchedulerName=self.scheduler_name
        )

        # Wait until stopped:
        if wait:
            self._poll_event(stop_scheduler_response['Status'], 'STOPPING')
        
    def delete(self):
        """
        Delete the current inference scheduler
        
        RETURNS
        =======
            delete_scheduler_response: dict
                A JSON dictionary with the response from the delete request API
        """
        if self.get_status() == 'STOPPED':
            delete_scheduler_response = self.client.delete_inference_scheduler(
                InferenceSchedulerName=self.scheduler_name
            )
            
        else:
            raise Exception('Scheduler must be stopped to be deleted.')
        
        return delete_scheduler_response
    
    def get_status(self):
        """
        Get current status of the inference scheduler
        
        RETURNS
        =======
            status: string
                The status of the inference scheduler, as extracted from the
                DescribeInferenceScheduler API
        """
        describe_scheduler_response = self.client.describe_inference_scheduler(
            InferenceSchedulerName=self.scheduler_name
        )
        status = describe_scheduler_response['Status']
        
        return status
    
    def list_inference_executions(self, 
                                  execution_status=None, 
                                  start_time=None, 
                                  end_time=None, 
                                  max_results=50):
        """
        This method lists all the past inference execution triggered by the
        current scheduler.
        
        PARAMS
        ======
            execution_status: string (default: None)
                Only keep the executions with a given status
                
            start_time: pandas.DateTime (default: None)
                Filters out the executions that happened before start_time
                
            end_time: pandas.DateTime (default: None)
                Filters out the executions that happened after end_time
                
            
            max_results: integer (default: 50)
                Max number of results you want to get out of this method
        
        RETURNS
        =======
            results_df: list of dict
                A list of all past inference executions, with each inference
                attributes stored in a python dictionary
        """
        # Built the execution request object:
        list_executions_request = {"MaxResults": max_results}
        list_executions_request["InferenceSchedulerName"] = self.scheduler_name
        if execution_status is not None:
            list_executions_request["Status"] = execution_status
        if start_time is not None:
            list_executions_request['DataStartTimeAfter'] = start_time
        if end_time is not None:
            list_executions_request['DataEndTimeBefore'] = end_time

        # Loops through all the inference executed by the current scheduler:
        has_more_records = True
        list_executions = []
        while has_more_records:
            list_executions_response = self.client.list_inference_executions(
                **list_executions_request
            )
            if "NextToken" in list_executions_response:
                list_executions_request["NextToken"] = list_executions_response["NextToken"]
            else:
                has_more_records = False

            list_executions = list_executions + \
                              list_executions_response["InferenceExecutionSummaries"]

        # Returns all the summaries in a list:
        self.execution_summaries = list_executions
        return list_executions
    
    def get_predictions(self):
        """
        This method loops through all the inference executions and build a
        dataframe with all the predictions generated by the model
        
        RETURNS
        =======
            results_df: pandas.DataFrame
                A dataframe with one prediction by row (1 for an anomaly or 0
                otherwise). Each row is indexed by timestamp.
        """
        # Fetch the list of execution summaries if there were not queried yet
        if self.execution_summaries is None:
            _ = self.list_inference_executions()
            
        # Loops through the executions summaries:
        results_df = []
        for execution_summary in self.execution_summaries:
            bucket = execution_summary['CustomerResultObject']['Bucket']
            key = execution_summary['CustomerResultObject']['Key']
            fname = f's3://{bucket}/{key}'
            results_df.append(pd.read_csv(fname, header=None))

        # Assembles them into a DataFrame:
        results_df = pd.concat(results_df, axis='index')
        results_df.columns = ['Timestamp', 'Predictions']
        results_df['Timestamp'] = pd.to_datetime(results_df['Timestamp'])
        results_df = results_df.set_index('Timestamp')
        
        return results_df
        
class LookoutEquipmentDataset:
    """
    A class to manage Lookout for Equipment datasets
    
    ATTRIBUTES
    ==========
        dataset_name: string
            The name of the dataset
        
        dataset_schema: string
            A JSON-formatted string describing the data schema the dataset
            must conform to.
            
    METHODS
    =======
        create():
            Create a dataset
            
        delete(force_delete):
            Delete the dataset
            
        ingest_data(bucket, prefix):
            Ingest S3 data into S3
    """
    def __init__(self, 
                 dataset_name, 
                 component_fields_map, 
                 region_name, 
                 access_role_arn):
        """
        Create a new instance to configure all the attributes necessary to 
        manage a Lookout for Equipment dataset.
        
        PARAMS
        ======
            dataset_name: string
                The name of the dataset to manage
                
            component_fields_map: string
                The mapping of the different fields associated to this dataset
                
            region_name: string
                The AWS region where the dataset is located
                
            access_role_arn: string
                The ARN of a role that will allow Lookout for Equipment to
                read data from the data source bucket on S3
        """
        self.dataset_name = dataset_name
        self.dataset_schema = create_data_schema(component_fields_map)
        self.client = get_client(region_name=region_name)
        self.role_arn = access_role_arn
    
    def create(self):
        """
        Creates a Lookout for Equipment dataset
        
        RETURNS
        =======
            create_dataset_response: string
                Response of the create dataset API
        """
        # Initialization:
        has_more_records = True
        # pp = pprint.PrettyPrinter(depth=4)
    
        # Checks if the dataset already exists:
        list_datasets_response = self.client.list_datasets(
            DatasetNameBeginsWith=self.dataset_name
        )

        dataset_exists = False
        for dataset_summary in list_datasets_response['DatasetSummaries']:
            if dataset_summary['DatasetName'] == self.dataset_name:
                dataset_exists = True
                break
    
        # If the dataset exists we just returns that message:
        if dataset_exists:
            print((f'Dataset "{self.dataset_name}" already exists and can be '
                    'used to ingest data or train a model.'))
    
        # Otherwise, we create it:
        else:
            print(f'Dataset "{self.dataset_name}" does not exist, creating it...\n')
    
            try:
                create_dataset_response = self.client.create_dataset(
                    DatasetName=self.dataset_name,
                    DatasetSchema={
                        'InlineDataSchema': self.dataset_schema
                    },
                    ClientToken=uuid.uuid4().hex
                )
                
                return create_dataset_response

            except Exception as e:
                print(e)
    
    def delete(self, force_delete=True):
        """
        Delete the dataset
        
        PARAMS
        ======
            force_delete: boolean (default: True)
                If set to True, also delete all the models that are using this
                dataset before deleting it. Otherwise, this method will list
                the attached models.
        """
        # Let's try to delete this dataset:
        try:
            delete_dataset_response = self.client.delete_dataset(
                DatasetName=self.dataset_name
            )
            print(f'Dataset "{self.dataset_name}" is deleted successfully.')
            return delete_dataset_response
            
        # This might not work if the dataset 
        # is used by an existing trained model:
        except Exception as e:
            error_code = e.response['Error']['Code']
            # If the dataset is used by existing models and we asked a
            # forced delete, we also delete the associated models before
            # trying again the dataset deletion:
            if (error_code == 'ConflictException') and (force_delete):
                print(('Dataset is used by at least a model, deleting the '
                       'associated model(s) before deleting dataset.'))
                models_list = list_models_for_datasets(
                    dataset_name_prefix=self.dataset_name
                )
    
                # List models that use this dataset and delete them:
                for model_to_delete in models_list:
                    self.client.delete_model(ModelName=model_to_delete)
                    print(f'- Model "{model_to_delete}" is deleted successfully.')
                    
                # Retry the dataset deletion
                delete_dataset_response = self.client.delete_dataset(
                    DatasetName=self.dataset_name
                )
                print(f'Dataset "{self.dataset_name}" is deleted successfully.')
                return delete_dataset_response
                
            # If force_delete is set to False, then we only list the models
            # using this dataset back to the user:
            elif force_delete == False:
                print('Dataset is used by the following models:')
                models_list = list_models_for_datasets(
                    dataset_name_prefix=self.dataset_name,
                )
    
                for model_name in models_list:
                    print(f'- {model_name}')
                    
                print(('Rerun this method with `force_delete` set '
                       'to True to delete these models'))
    
            # Dataset name not found:
            elif (error_code == 'ResourceNotFoundException'):
                print((f'Dataset "{self.dataset_name}" not found: creating a '
                        'dataset with this name is possible.'))
    
    def ingest_data(self, bucket, prefix):
        """
        Ingest data from an S3 location into the dataset
        
        PARAMS
        ======
            bucket: string
                Bucket name where the data to ingest are located
                
            prefix: string
                Actual location inside the aforementioned bucket
        
        RETURNS
        =======
            start_data_ingestion_job_response: string
                Response of the start ingestion job API call
        """
        # Configure the input location:
        ingestion_input_config = dict()
        ingestion_input_config['S3InputConfiguration'] = dict([
            ('Bucket', bucket),
            ('Prefix', prefix)
        ])

        # Start data ingestion:
        start_data_ingestion_job_response = self.client.start_data_ingestion_job(
            DatasetName=self.dataset_name,
            RoleArn=self.role_arn, 
            IngestionInputConfiguration=ingestion_input_config,
            ClientToken=uuid.uuid4().hex
        )
        
        return start_data_ingestion_job_response
        
class LookoutEquipmentModel:
    """
    A class to manage Lookout for Equipment models
    
    ATTRIBUTES
    ==========
        dataset_name: string
            The name of the dataset
        
        model_name: string
            The name of the model
            
        create_model_request: dict
            The parameters to be used to train the model
            
    METHODS
    =======
        set_label_data(bucket, prefix, access_role_arn):
            Tell Lookout for Equipment to look for labelled data and where to
            find them on S3
            
        set_target_sampling_rate(sampling_rate):
            Set the sampling rate to use before training the model
            
        set_time_periods():
            Set the training / evaluation time split
            
        set_subset_schema(subset_schema):
            Configure the inline data schema that will let Lookout for Equipment
            knows that it needs to select a subset of all the signals configured
            at ingestion
            
        train():
            Train the model as configured with this object
            
        poll_model_training(sleep_time):
            This function polls the model describe API and print a status until 
            the training is done.

        delete():
            Delete the current model
    """
    def __init__(self, model_name, dataset_name, region_name):
        """
        """
        self.client = get_client(region_name=region_name)
        self.model_name = model_name
        self.create_model_request = dict()
        self.create_model_request.update({
            'ModelName': model_name,
            'DatasetName': dataset_name
        })
        
    def set_label_data(self, bucket, prefix, access_role_arn):
        """
        Tell Lookout for Equipment to look for labelled data and where to
        find them on S3
        
        PARAMS
        ======
            bucket: string
                Bucket name where the labelled data can be found
                
            prefix: string
                Prefix where the labelled data can be found
                
            access_role_arn: string
                A role that Lookout for Equipment can use to access the bucket
                and prefix aforementioned
        """
        labels_input_config = dict()
        labels_input_config['S3InputConfiguration'] = dict([
            ('Bucket', bucket),
            ('Prefix', prefix)
        ])
        self.create_model_request.update({
            'RoleArn': access_role_arn,
            'LabelsInputConfiguration': labels_input_config
        })
        
    def set_target_sampling_rate(self, sampling_rate):
        """
        Set the sampling rate to use before training the model
        
        PARAMS
        ======
            sampling_rate: string
                One of [PT1M, PT5S, PT15M, PT1S, PT10M, PT15S, PT30M, PT10S, 
                PT30S, PT1H, PT5M]
        """
        self.create_model_request.update({
            'DataPreProcessingConfiguration': {
                'TargetSamplingRate': sampling_rate
            },
        })
    
    def set_time_periods(self, 
                         evaluation_start, 
                         evaluation_end, 
                         training_start, 
                         training_end):
        """
        Set the training / evaluation time split
        
        PARAMS
        ======
            evaluation_start: datetime
                Start of the evaluation period

            evaluation_end: datetime
                End of the evaluation period

            training_start: datetime
                Start of the training period

            training_end: datetime
                End of the training period
        """
        self.create_model_request.update({
            'TrainingDataStartTime': training_start.to_pydatetime(),
            'TrainingDataEndTime': training_end.to_pydatetime(),
            'EvaluationDataStartTime': evaluation_start.to_pydatetime(),
            'EvaluationDataEndTime': evaluation_end.to_pydatetime()
        })
        
    def set_subset_schema(self, subset_schema):
        """
        Configure the inline data schema that will let Lookout for Equipment
        knows that it needs to select a subset of all the signals configured
        at ingestion
        
        PARAMS
        ======
            subset_schema: string
                A JSON string describing which signals to keep for this model
        """
        data_schema_for_model = {
            'InlineDataSchema': create_data_schema(subset_schema),
        }
        self.create_model_request['DatasetSchema'] = data_schema_for_model

    def train(self):
        """
        Train the model as configured with this object
        
        RETURNS
        =======
            create_model_response: string
                The create model API response in JSON format
        """
        self.create_model_request.update({
            'ClientToken': uuid.uuid4().hex
        })
        
        create_model_response = self.client.create_model(
            **self.create_model_request
        )
        
        return create_model_response
        
    def delete(self):
        """
        Delete the current model

        RETURNS
        =======
            delete_model_response: string
                The delete model API response in JSON format
        """
        try:
            delete_model_response = self.client.delete_model(
                ModelName=self.model_name
            )
            return delete_model_response
            
        except Exception as e:
            error_code = e.response['Error']['Code']
            # If the dataset is used by existing models and we asked a
            # forced delete, we also delete the associated models before
            # trying again the dataset deletion:
            if (error_code == 'ConflictException'):
                print(('Model is currently being used (a training might be in '
                       'progress. Wait for the process to be completed and '
                       'retry.'))

    def poll_model_training(self, sleep_time=60):
        """
        This function polls the model describe API and print a status until the
        training is done.
        
        PARAMS
        ======
            sleep_time: integer (default: 60)
                How many seconds should we wait before polling again
        """
        describe_model_response = self.client.describe_model(
            ModelName=self.model_name
        )
        
        status = describe_model_response['Status']
        while status == 'IN_PROGRESS':
            time.sleep(sleep_time)
            describe_model_response = self.client.describe_model(
                ModelName=self.model_name
            )
            status = describe_model_response['Status']
            print(
                str(pd.to_datetime(datetime.now()))[:19],
                "| Model training:", 
                status
            )