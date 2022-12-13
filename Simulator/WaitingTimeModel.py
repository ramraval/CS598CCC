from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.1

"""
Class definition for waiting time model used to dynamically predict waiting time for submitted jobs
"""


class WaitingTimeModel:
    def __init__(self, training_trace, waiting_time_threshold_sec: int, dataset: str):
        self.training_trace = training_trace
        self.waiting_time_threshold_sec = waiting_time_threshold_sec
        self.dataset = dataset
        self.random_state = 0
        self.max_features = 'sqrt'

        if self.dataset == 'ANL':
            self.n_estimators = 250
            self.min_samples_split = 100
            self.min_samples_leaf = 250
        else:
            self.n_estimators = 100
            self.min_samples_split = 500
            self.min_samples_leaf = 100

    """
    Train waiting time model on training trace given specified waiting time threshold; return model fit on training data
    """

    def train_waiting_time_model(self):
        self.training_trace['should_wait_waiting_time_actual'] = self.training_trace[
                                                                     'wait_time'] < self.waiting_time_threshold_sec

        X = self.training_trace[['user_id', 'submit_time', 'requested_time', 'requested_CPUs',
                                 'num_running_jobs', 'num_waiting_jobs',
                                 'running_job_requested_CPUs', 'running_job_requested_CPU_time',
                                 'running_job_mean_CPUs',
                                 'running_job_mean_CPU_time', 'running_job_requested_wallclock_limit',
                                 'running_job_mean_wallclock_limit',
                                 'waiting_job_requested_CPUs', 'waiting_job_requested_CPU_time',
                                 'waiting_job_mean_CPUs',
                                 'waiting_job_mean_CPU_time', 'waiting_job_requested_wallclock_limit',
                                 'waiting_job_mean_wallclock_limit',
                                 'elapsed_runtime_total', 'elapsed_runtime_mean', 'elapsed_waiting_time_total',
                                 'elapsed_waiting_time_mean']]
        y = self.training_trace['should_wait_waiting_time_actual']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

        clf = GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=self.random_state,
                                         max_features=self.max_features, min_samples_leaf=self.min_samples_leaf,
                                         min_samples_split=self.min_samples_split)

        clf.fit(X_train, y_train.values.ravel())
        return clf
