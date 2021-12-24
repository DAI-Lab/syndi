import os
import shutil
import unittest

import pytest

import syndi.task as task
import syndi.task_evaluator as task_evaluator

_BASE_PATH = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.usefixtures("change_test_dir")
class TestTaskEvauator(unittest.TestCase):
    def setUp(self):
        # output directory setup
        output_dir = "tasks"
        task_id = "test_id"
        task_output_dir = os.path.join(output_dir, task_id)
        if os.path.exists(task_output_dir):
            shutil.rmtree(task_output_dir)
        os.mkdir(task_output_dir)
        self.output_dir = task_output_dir

    def test_baseline(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Attrition",
                              path_to_generator=path_to_generator,
                              sampling_method_id="baseline", pycaret_model="lr",
                              run_num=1, output_dir=self.output_dir)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'baseline')
        expected_synth_data_path = os.path.join(test_task.output_dir, "classifier_lr.pkl")
        self.assertTrue(os.path.exists(expected_synth_data_path))

    def test_sampling_method_original_0_2(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Attrition",
                              path_to_generator=path_to_generator,
                              sampling_method_id="original_0", pycaret_model="lr",
                              run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 20/60')

    def test_sampling_method_original_1_2(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Attrition",
                              path_to_generator=path_to_generator,
                              sampling_method_id="original_1", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 40/60')

    def test_sampling_method_original_2_2(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Attrition",
                              path_to_generator=path_to_generator,
                              sampling_method_id="original_2", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'original 60/60')

    def test_sampling_method_uniform(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id=1, train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Attrition",
                              path_to_generator=path_to_generator,
                              sampling_method_id="uniform", pycaret_model="lr", run_num=1)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        result = evaluator.evaluate_task()
        sampling_methods = result[3]
        self.assertEqual(sampling_methods, 'uniform')

    def test_get_scores(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id="36_none_baseline_svm_0",
                              train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Attrition",
                              path_to_generator=path_to_generator,
                              sampling_method_id="baseline", pycaret_model="lr", run_num=0)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        df = evaluator.train_data
        for i in range(10):
            df2 = {column: i for column in df.columns}
            df2[df.columns[-1]] = "Maybe"
            df = df.append(df2, ignore_index=True)
        evaluator.train_data = df
        result = evaluator.evaluate_task()
        self.assertTrue(not result[0] is None)

    def test_svm_classifier(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id="36_none_baseline_svm_0", train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Attrition",
                              path_to_generator=path_to_generator,
                              sampling_method_id="baseline", pycaret_model="svm", run_num=0)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        evaluator.evaluate_task()

    def test_uniform_regression(self):
        path_to_generator = "generators/default_gaussain_copula.pkl"
        test_task = task.Task(task_id="36_none_baseline_lr_0", train_dataset="data/train.csv",
                              test_dataset="data/test.csv", target="Age",
                              path_to_generator=path_to_generator,
                              sampling_method_id="uniform", pycaret_model="lr", run_num=0,
                              is_regression=True)
        evaluator = task_evaluator.Task_Evaluator(test_task)
        evaluator.evaluate_task()

    def test_uniform_regression_all_state_dataset(self):
        task_output_path = "tasks/"
        path_to_generators = "regression_generators/"
        tasks = task.create_tasks(train_dataset="regression_data/train.csv",
                                  test_dataset="regression_data/test.csv", target="charges",
                                  path_to_generators=path_to_generators, pycaret_models=[
                                      "lr", "ridge", "kr"],
                                  task_sampling_method="uniform", run_num=1,
                                  output_dir=task_output_path,
                                  is_regression=True)
        test_task = tasks[0]
        # run benchmark on tasks
        evaluator = task_evaluator.Task_Evaluator(test_task)
        evaluator.evaluate_task()


if __name__ == '__main__':
    unittest.main()
