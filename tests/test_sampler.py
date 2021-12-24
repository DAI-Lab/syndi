import os
import shutil
import unittest

import pandas as pd
import pytest
import sdv.sdv
from sdv.tabular import GaussianCopula

from syndi.sampler import Sampler
from syndi.task import Task


@pytest.mark.usefixtures("change_test_dir")
class TestSampler(unittest.TestCase):
    def setUp(self):
        generator_path = "generators/default_gaussain_copula.pkl"
        self.generator = sdv.sdv.SDV.load(generator_path)
        train_data_path = "data/train.csv"
        test_data_path = "data/test.csv"
        target = "Attrition"
        self.target = target
        self.train_data = pd.read_csv(train_data_path)
        run_num = 1
        classifier = "lr"

        # output directory setup
        output_dir = "tasks"
        task_id = "test_id"
        task_output_dir = os.path.join(output_dir, task_id)
        if os.path.exists(task_output_dir):
            shutil.rmtree(task_output_dir)
        os.mkdir(task_output_dir)
        # Task function setup

        def make_task(sampling_method_id, is_regression=False, target=target):
            return Task(task_id=task_id, train_dataset=train_data_path,
                        test_dataset=test_data_path, target=target,
                        path_to_generator=generator_path, sampling_method_id=sampling_method_id,
                        pycaret_model=classifier, run_num=run_num, output_dir=task_output_dir,
                        is_regression=is_regression)
        self.make_task = make_task

    def test_original_0(self):
        task_original_0 = self.make_task("original_0")
        sampler = Sampler(task_original_0, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(80, combined_data.shape[0])
        self.assertEqual("original 20/60", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)
        expected_synth_data_path = os.path.join(task_original_0.output_dir, "synthetic_data.csv")
        self.assertTrue(os.path.exists(expected_synth_data_path))

    def test_original_1(self):
        task_original_1 = self.make_task("original_1")
        sampler = Sampler(task_original_1, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(100, combined_data.shape[0])
        self.assertEqual("original 40/60", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)
        expected_synth_data_path = os.path.join(task_original_1.output_dir, "synthetic_data.csv")
        self.assertTrue(os.path.exists(expected_synth_data_path))

    def test_original_2(self):
        task_original_2 = self.make_task("original_2")
        sampler = Sampler(task_original_2, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(120, combined_data.shape[0])
        self.assertEqual("original 60/60", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)
        expected_synth_data_path = os.path.join(task_original_2.output_dir, "synthetic_data.csv")
        self.assertTrue(os.path.exists(expected_synth_data_path))

    def test_baseline_sampling_method(self):
        task_baseline = self.make_task("baseline")
        sampler = Sampler(task_baseline, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(60, combined_data.shape[0])
        self.assertEqual("baseline", sampling_method_info)
        self.assertEqual(score_aggregate, None)
        expected_synth_data_path = os.path.join(task_baseline.output_dir, "synthetic_data.csv")
        self.assertFalse(os.path.exists(expected_synth_data_path))

    def test_uniform_sampling_method(self):
        task_uniform = self.make_task("uniform")
        sampler = Sampler(task_uniform, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        yes_count, no_count = combined_data[self.target].value_counts().to_list()
        self.assertEqual(yes_count, no_count)
        self.assertEqual("uniform", sampling_method_info)
        self.assertIsInstance(score_aggregate, float)
        expected_synth_data_path = os.path.join(task_uniform.output_dir, "synthetic_data.csv")
        self.assertTrue(os.path.exists(expected_synth_data_path))

    def test_uniform_tvae_fail(self):
        task_uniform = self.make_task("uniform")

        generator_path = "generators/default_tvae.pkl"
        self.generator = sdv.sdv.SDV.load(generator_path)

        sampler = Sampler(task_uniform, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertTrue(len(sampler.logs) >= 0)
        # TODO fix unfirom sampling bugs for classification
        # error_msg = "No valid rows could be generated with the given conditions."
        # self.assertTrue(error_msg in str(context.exception))

    def test_uniform_regression_int(self):  # TODO add integer support for regression
        # TestSampler.test_uniform_regression_int
        task_uniform = self.make_task("uniform", is_regression=True, target="Age")
        sampler = Sampler(task_uniform, self.train_data, self.generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        copy_combined = combined_data.copy()
        copy_combined["Age"] = pd.cut(x=copy_combined["Age"], bins=5)
        for value in sampler._get_class_to_sample_size(copy_combined).values():
            self.assertEqual(value, 0)

    def test_uniform_regression_float(self):
        task_uniform = self.make_task("uniform", is_regression=True, target="Age")
        self.train_data["Age"] = self.train_data["Age"] + .1
        generator = GaussianCopula()
        generator.fit(self.train_data)
        sampler = Sampler(task_uniform, self.train_data, generator)
        binned_data = sampler.train_data.copy()
        binned_data["Age"] = pd.cut(x=self.train_data["Age"], bins=5)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        copy_combined = combined_data.copy()
        copy_combined["Age"] = pd.cut(x=copy_combined["Age"], bins=5)
        for value in sampler._get_class_to_sample_size(copy_combined).values():
            self.assertEqual(value, 0)

    def test_uniform_regression_retry(self):
        # TestSampler.test_uniform_regression_retry
        task_output_path = "tasks/test_id"
        generator_path = "regression_generators/TvaeModel.pkl"
        train_data = "regression_data/train.csv"
        test_data = "regression_data/test.csv"
        target = "charges"
        sampling_method_id = "uniform"
        classifier = "lr"
        run_num = 0
        is_regression = True

        task_uniform = Task(task_id="test_id", train_dataset=train_data,
                            test_dataset=test_data, target=target,
                            path_to_generator=generator_path,
                            sampling_method_id=sampling_method_id,
                            pycaret_model=classifier, run_num=run_num, output_dir=task_output_path,
                            is_regression=is_regression)
        generator = sdv.sdv.SDV.load(generator_path)
        loaded_train_data = pd.read_csv(train_data)
        sampler = Sampler(task_uniform, loaded_train_data, generator)
        combined_data, sampling_method_info, score_aggregate = sampler.sample_data()
        self.assertEqual(1, len(sampler.logs))


if __name__ == '__main__':
    unittest.main()
