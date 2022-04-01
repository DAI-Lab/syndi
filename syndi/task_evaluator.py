import os

import pandas as pd  # Basic data manipulation
import sdv.sdv
import sklearn
import numpy as np
from pycaret import classification, regression

from syndi.sampler import Sampler

CLASSIFICATION_METRICS = ['auc', 'f1', 'recall', 'precision', 'accuracy']
REGRESSION_METRICS = ["mae", "mse", "r2", "msle", "gini"]


class Task_Evaluator():
    def __init__(self, task):
        self.task = task
        self.train_data = pd.read_csv(task.train_dataset)
        self.test_data = pd.read_csv(task.test_dataset)
        generator = sdv.sdv.SDV.load(task.path_to_generator)
        self.sampler = Sampler(task, self.train_data, generator)

    def evaluate_task(self, metrics=None):
        """Run benchmark testing on a task. Save intermedia data, trained models, and optimized
        hyperparameters. Return testing results.

        Args:
            task (Task):
                a task instance storing meta information of the task.
            metrics (list)
                a list of strings to identify the metric functions.
            output_path (str):
                a directory path to store the intermedia data, model and hyperparametes.
            agnostic_metrics (boolean):
                whether to record dataset agnostic metrics in results

        Returns:
            list:
                benchmarking results of each run.
        """
        all_metrics = REGRESSION_METRICS if self.task.is_regression else CLASSIFICATION_METRICS
        if metrics is None:
            metrics = all_metrics

        combined_data, sampling_method_info, score_aggregate = self.sampler.sample_data()

        predictions = self._regression(
            combined_data) if self.task.is_regression else self._classify(combined_data)
        ground_truth = predictions[self.task.target]
        output_predictions = predictions["Label"]
        output_score = None
        if "Score" in predictions.columns:
            output_score = predictions["Score"]
        scores = self._get_scores(ground_truth, output_predictions, output_score)
        # make dictionary of metric name to score

        metric_to_score = {metric: score for metric, score in zip(all_metrics, scores)}
        # record entry
        row = [self.task.task_id, self.task.path_to_generator, self.task.pycaret_model,
               sampling_method_info, self.task.run_num]
        for metric in metrics:
            row += [metric_to_score[metric]]  # TODO change to append
        return row

    def get_sampler_logs(self):
        return self.sampler.logs

    def _get_scores(self, ground_truth, predictions, scores):
        if self.task.is_regression:
            return self._get_regression_scores(ground_truth, predictions, scores)
        else:
            return self._get_classification_scores(ground_truth, predictions, scores)

    @classmethod
    def _get_classification_scores(cls, ground_truth, classifier_predictions, classifier_score):
        labels = sorted(ground_truth.unique())

        precision_avg, recall_avg, f1_avg, _ = sklearn.metrics.precision_recall_fscore_support(
            ground_truth, classifier_predictions, average="macro", labels=labels)
        precision_label, recall_label, f1_label, support =\
            sklearn.metrics.precision_recall_fscore_support(
                ground_truth, classifier_predictions, labels=labels)
        accuracy = sklearn.metrics.accuracy_score(ground_truth, classifier_predictions)
        auc = None
        if classifier_score is not None:
            auc = sklearn.metrics.roc_auc_score(
                ground_truth, classifier_score, average='macro', multi_class="ovr")

        # def convert_labels_lists_to_dict(label_scores):
        #     scores = {label:score for label, score in zip(labels,label_scores)}
        #     return scores

        precision = precision_avg  # label specific: convert_labels_lists_to_dict(precision_label)
        recall = recall_avg  # label specific: convert_labels_lists_to_dict(recall_label)
        f1 = f1_avg  # label convert_labels_lists_to_dict(f1_label)
        # support = convert_labels_lists_to_dict(support)

        return [auc, f1, recall, precision, accuracy, support]
    @classmethod
    def get_gini(cls, data, pred, actual, weight_var=None):
        lr = "lr"
        data_size = data.shape[0]
        weights = np.ones(data_size)
        if weight_var is not None:
            weights = data[weight_var].to_numpy()
            weights = weights
        else:
            weight_var = "weights"
        frame = {
            lr: round(data[pred]/weights,8),
            pred: data[pred], 
            actual: data[actual]/np.sum(data[actual]),
            weight_var: weights/np.sum(weights)
            }
        df = pd.DataFrame(frame)
        # take mean of rows with same lr
        df = df.groupby(lr).mean().reset_index() 
        # sort by lr
        df = df.sort_values(lr, ascending=True)
        #calculate cumulative sums
        cum_weights = np.cumsum(df[weight_var]) / np.sum(df[weight_var])
        cum_actual = np.cumsum(df[actual])/np.sum(df[actual])
        #calculate A
        A = np.trapz(y=cum_weights - cum_actual, x=cum_weights)
        #calculate A+B
        df = df.copy().sort_values(actual, ascending=True)
        perfect_cum_actual = np.cumsum(df[actual])/np.sum(df[actual])
        perfect_weights = df[weight_var]
        perfect_normalized_weights = perfect_weights/ np.sum(perfect_weights)
        A_B = .5-np.sum(perfect_cum_actual*perfect_normalized_weights)
        #calculate gini
        gini = A/A_B
        return gini

    @classmethod
    def _get_regression_scores(cls, ground_truth, predictions, classifier_score):
        mae = sklearn.metrics.mean_absolute_error(ground_truth, predictions)
        mse = sklearn.metrics.mean_squared_error(ground_truth, predictions)
        r2 = sklearn.metrics.r2_score(ground_truth, predictions)
        msle = None
        try:
            msle = sklearn.metrics.mean_squared_log_error(ground_truth, predictions)
        except ValueError:
            pass
        actual="ground_truth"
        pred="predictions"
        gini_input_dict = {
            pred:predictions,
            actual:ground_truth,
        }
        gini_input_data = pd.DataFrame.from_dict(gini_input_dict)
        gini = Task_Evaluator.get_gini(gini_input_data, pred, actual)
        return [mae, mse, r2, msle, gini]

    def _store_classifier(self, classifier_model):
        task_output_dir = self.task.output_dir
        classifier_file_name = "classifier_{}".format(self.task.pycaret_model)
        classifier_output_path = os.path.join(task_output_dir, classifier_file_name)
        classification.save_model(classifier_model, classifier_output_path)

    def _store_regresser(self, regression_model):
        task_output_dir = self.task.output_dir
        regressor_file_name = "regressor_{}".format(self.task.pycaret_model)
        regressor_output_path = os.path.join(task_output_dir, regressor_file_name)
        regression.save_model(regression_model, regressor_output_path)

    def _classify(self, combined_data):
        # TODO, check for ordinal and categorical features
        self._classifier_setup(combined_data)
        # Train classifier
        classifier = classification.create_model(self.task.pycaret_model, verbose=False)
        # Store Classifier
        if self.task.output_dir:
            self._store_classifier(classifier)
        # Predict on Test set
        predictions = classification.predict_model(
            classifier, verbose=False)  # TODO get raw_scores for AUC
        return predictions

    def _classifier_setup(self, combined_data):
        classification.setup(combined_data.sample(frac=1),  # shuffles the data
                             target=self.task.target,
                             test_data=self.test_data,
                             fold_strategy="kfold",  # TODO allow more strategies as hyperparam
                             silent=True,
                             verbose=False)

    def _regression(self, combined_data):
        self._regression_setup(combined_data)
        # Train
        regresser = regression.create_model(self.task.pycaret_model, verbose=False)
        # Store
        if self.task.output_dir:
            self._store_regresser(regresser)
        # Predict on Test set
        predictions = regression.predict_model(
            regresser, verbose=False)  # TODO get raw_scores for AUC
        return predictions

    def _regression_setup(self, combined_data):
        regression.setup(combined_data.sample(frac=1),  # shuffles the data
                         target=self.task.target,
                         test_data=self.test_data,
                         fold_strategy="kfold",  # TODO allow more strategies as hyperparam
                         silent=True,
                         verbose=False)
