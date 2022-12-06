"""
Copyright (c) 2020 <Sustainable Computing Lab>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
import os
import sys
import numpy as np
import pandas as pd

from collections import deque
from enum import Enum
from typing import Dict, List, Optional
from sortedcontainers import SortedList
from node import Node
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

UNIT_CPU_COST_HR = 0.048  # Cost Estimation assuming m5 family
RESERVE_DISCOUNT = 0.6  # 3 year discount
SECONDS_IN_HOUR = 3600
TEST_SIZE = 0.1
RUNTIME_THRESHOLD_VALUES = [15]
WAITING_TIME_THRESHOLD_VALUES = [6]

"""
Class definition for event type enum
"""


class etype(Enum):
    schedule = 1
    finish = 2


"""
Class definition for event;
Event type can be "Start" indicated by 1 or "Expire" indicated by 2
"""


class Event:
    def __init__(self, etype: etype, time: int, job_id: int):
        self.etype = etype
        self.time = time
        self.job_id = job_id


"""
Class definition for waiting time model used to dynamically predict waiting time for submitted jobs
"""

class WaitingTimeModel:
    def __init__(self, training_trace: str, waiting_time_threshold_sec: int, dataset: str):
        self.training_trace = training_trace
        self.waiting_time_threshold_sec = waiting_time_threshold_sec
        self.dataset = dataset

    """
    Train waiting time model on training trace given specified waiting time threshold; return model fit on training data
    """

    def train_waiting_time_model(self):
        self.training_trace['should_wait_waiting_time_actual'] = self.training_trace['wait_time'] < self.waiting_time_threshold_sec

        X = self.training_trace[['user_id', 'submit_time', 'requested_time', 'requested_CPUs',
                'num_running_jobs', 'num_waiting_jobs',
                'running_job_requested_CPUs', 'running_job_requested_CPU_time', 'running_job_mean_CPUs',
                'running_job_mean_CPU_time', 'running_job_requested_wallclock_limit',
                'running_job_mean_wallclock_limit',
                'waiting_job_requested_CPUs', 'waiting_job_requested_CPU_time', 'waiting_job_mean_CPUs',
                'waiting_job_mean_CPU_time', 'waiting_job_requested_wallclock_limit',
                'waiting_job_mean_wallclock_limit',
                'elapsed_runtime_total', 'elapsed_runtime_mean', 'elapsed_waiting_time_total',
                'elapsed_waiting_time_mean']]
        y = self.training_trace['should_wait_waiting_time_actual']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

        if self.dataset == 'ANL':
            clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=0)

        clf.fit(X_train, y_train.values.ravel())
        return clf

## Helper functions

def division_helper(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0

def get_job_attributes(job):
    job_id, time, cpu_alloc, runtime = job.job_id, job.submit_time, job.allocated_CPUs, job.runtime
    return job_id, time, cpu_alloc, runtime

def is_long_job(job, runtime_threshold) -> bool:
    return job['should_wait_runtime_predicted_' + str(runtime_threshold)]

"""
Class definition for simulator

Instance Variables:
dataset: String; input dataset (ANL or RICC)
NUM_VMS: int; Number of VMs/nodes in cluster
VM_CPU: int; number of CPU cores per VM
input_trace: Dataframe; Input trace as pandas data frame
training_trace: Dataframe; trace used to train waiting time model

next_jobID: int; jobID of next job arrival
current_reserve_jobs: SortedList(tuple); tuple - (finishtime, jobID, nodeID)
job_schedule_map: Dictionary; key - jobID, value - 'R' or 'D'
wait_time_map: Dictionary; key - jobID, value - int
wait_queue: deque; waiting queue
"""


class Simulator:
    def __init__(
            self,
            dataset: str,
            NUM_VMS: int,
            VM_CPU: int,
            input_trace: pd.DataFrame,
            training_trace: pd.DataFrame
    ) -> None:
        self.dataset = dataset
        self.NUM_VMS = NUM_VMS
        self.VM_CPU = VM_CPU

        self.input_trace = input_trace
        self.training_trace = training_trace
        self.job_count = input_trace.job_id.count()

        self.next_jobID: int = 0
        self.current_reserve_jobs: SortedList = SortedList()
        self.job_schedule_map: Dict[int, str] = {}
        self.wait_time_map: Dict[int, int] = {}
        self.wait_queue: deque = deque()

        self.mean_waiting_time: float
        self.reserved_cost: float
        self.on_demand_cost: float

        # Cluster
        self._reserve_nodes: List[Node] = []
        vm_name = "m5.16xlarge"
        for i in range(NUM_VMS):
            n = Node(vm_name, i, VM_CPU)
            self._reserve_nodes.append(n)

    """
    Determine which fixed nodes to schedule jobs on
    If sufficient fixed nodes are unavailable, return an empty dict (must queue or run on on-demand)
    Otherwise, return dict indicating how many CPUs to reserve on which nodes 
    """

    def _can_schedule(self, cpu_req: int) -> Optional[dict]:
        node_dic = {}
        for node in self._reserve_nodes:
            cpus_scheduled = min(cpu_req, node.idle_cpu)
            if cpus_scheduled:
                node_dic[node._nodeID] = cpus_scheduled
                cpu_req -= cpus_scheduled

        return node_dic if cpu_req == 0 else None

    """
    Reserve specified CPUs on specified nodes for a given job
    Add given job to set of currently running jobs
    """

    def _schedule(
            self,
            schedule_dict: dict,
            start_time: int,
            runtime_sec: int,
            job_id: int,
    ) -> None:
        for node_id, CPU_count in schedule_dict.items():
            self._reserve_nodes[node_id].schedule_job(CPU_count, job_id)
        finish_time = start_time + runtime_sec
        self.current_reserve_jobs.add((finish_time, start_time, job_id, schedule_dict.keys()))

    """
    Returns the next event
    """

    def _get_next_event(self) -> Optional[Event]:
        next_arr: pd.Series = None
        next_dep: pd.Series = None
        next_arr_time = float('inf')
        next_dep_time = float('inf')

        # Next Arrival
        if self.next_jobID < self.job_count:
            next_arr = self.input_trace.loc[self.input_trace['job_id'] == self.next_jobID].iloc[0]
            next_arr_time = next_arr.submit_time

        # Next Departure
        if self.current_reserve_jobs:
            next_dep_time, _, dep_job_id, _ = self.current_reserve_jobs[0]
            next_dep = self.input_trace.loc[self.input_trace['job_id'] == dep_job_id].iloc[0]

        # All the events are played (0, 0)
        if next_arr is None and next_dep is None:
            return None

        # Compare time events
        if next_dep_time <= next_arr_time:
            next_event = Event(etype.finish, next_dep_time, next_dep.job_id)
            # Add back the capacity
            _, _, _, node_ids = self.current_reserve_jobs.pop(0)
            for node_id in node_ids:
                self._reserve_nodes[node_id].remove_job(next_dep.job_id)
        else:
            next_event = Event(etype.schedule, next_arr_time, next_arr.job_id)
            self.next_jobID += 1

        return next_event

    def is_short_wait(self, model, cur_job) -> bool:
        X_test = self.calc_system_state_features(cur_job)
        return model.predict(X_test)

    def queue_is_empty(self) -> bool:
        return not self.wait_queue

    def set_waiting_time(self, job, waiting_time):
        self.wait_time_map[job.job_id] = waiting_time

    def calc_on_demand_cost_for_job(self, job, runtime):
        CPUs_needed = self.VM_CPU * math.ceil(job.allocated_CPUs / self.VM_CPU)
        return CPUs_needed * (runtime / SECONDS_IN_HOUR) * UNIT_CPU_COST_HR

    """
    Calculates approximate cost of running fixed cluster for trace duration
    """
    def _compute_fixed_cost(self) -> float:
        self.input_trace['submit_time'] = pd.to_numeric(self.input_trace['submit_time'])
        self.input_trace['end_time'] = pd.to_numeric(self.input_trace['end_time'])

        start_time = self.input_trace.loc[self.input_trace['submit_time'].idxmin()].submit_time
        end_time = self.input_trace.loc[self.input_trace['end_time'].idxmax()].end_time
        hours_in_operation = (end_time - start_time) / SECONDS_IN_HOUR
        return hours_in_operation * UNIT_CPU_COST_HR * self.NUM_VMS * self.VM_CPU * (1 - RESERVE_DISCOUNT)

    def calc_waiting_time_and_reserved_cost(self):
        self.input_trace["wait_time_sec"] = (self.input_trace["job_id"].map(self.wait_time_map))
        self.mean_waiting_time, self.reserved_cost = self.input_trace.wait_time_sec.mean(), self._compute_fixed_cost()

    """
    Check for finished jobs on reserved nodes given current time.
    Add the back the resources to the respective nodes.
    """

    def _check_expired_jobs(self, current_time: int) -> None:
        while self.current_reserve_jobs:
            finish_time, _, job_id, node_ids = self.current_reserve_jobs[0]
            if finish_time <= current_time:
                for node_id in node_ids:
                    # Remove the job from the node
                    self._reserve_nodes[node_id].remove_job(job_id)
                # Pop the job from the running job list
                self.current_reserve_jobs.pop(0)
            else:
                return


    def resubmit_long_jobs_after_spec_execution(self, job, runtime_threshold_sec):
        # Resubmit job after running for "runtime_threshold_sec" time per speculative execution
        job.loc['speculative_execution'] = True
        job.loc['submit_time'] = job['submit_time'] + runtime_threshold_sec

        # Place job in correct place in trace
        temp_df = self.input_trace[self.input_trace['submit_time'] > job.submit_time]
        new_index = temp_df.iloc[0].job_id
        job.loc['job_id'] = new_index

        job = job.to_frame().T

        # Reorder and re-index input trace to reflect new timing of job
        self.input_trace = pd.concat([self.input_trace.iloc[:new_index], job,
                                      self.input_trace.iloc[new_index:]]).reset_index(drop=True)
        self.input_trace['job_id'] = self.input_trace.index

    def print_results(self) -> None:
        print(
            "Reserved cost is ",
            self.reserved_cost,
            ", On demand cost is ",
            self.on_demand_cost,
            ", Total cost is ",
            self.reserved_cost + self.on_demand_cost,
            ", Mean waiting time is ",
            self.mean_waiting_time / 60, " minutes"
        )

    """
    Calculate system state upon submission of a job in order to predict waiting time
    """
    def calc_system_state_features(self, waiting_job):
        starting_times = [job[1] for job in self.current_reserve_jobs]

        running_job_ids = [job[2] for job in self.current_reserve_jobs]
        running_jobs = self.input_trace.loc[self.input_trace['job_id'].isin(running_job_ids)].index
        waiting_jobs = self.input_trace.loc[self.input_trace['job_id'].isin(self.wait_queue)].index

        user_id = waiting_job.user_id
        submit_time = waiting_job.submit_time
        requested_time = waiting_job.requested_time
        requested_CPUs = waiting_job.requested_CPUs
        elapsed_runtimes = [submit_time - start_time for start_time in starting_times]

        num_running_jobs = len(self.current_reserve_jobs)
        num_waiting_jobs = len(self.wait_queue)

        running_job_total_requested_CPUs = self.input_trace.loc[running_jobs, 'requested_CPUs'].sum()
        running_job_mean_requested_CPUs = division_helper(running_job_total_requested_CPUs, num_running_jobs)

        running_job_total_requested_CPU_time = self.input_trace.loc[running_jobs, 'CPU_time'].sum()
        running_job_mean_requested_CPU_time = division_helper(running_job_total_requested_CPU_time,
                                                              num_running_jobs)

        running_job_total_requested_wallclock_time = self.input_trace.loc[running_jobs, 'requested_time'].sum()
        running_job_mean_requested_wallclock_time = division_helper(running_job_total_requested_wallclock_time,
                                                                    num_running_jobs)

        waiting_job_total_requested_CPUs = self.input_trace.loc[waiting_jobs, 'requested_CPUs'].sum()
        waiting_job_mean_requested_CPUs = division_helper(waiting_job_total_requested_CPUs, num_waiting_jobs)

        waiting_job_total_requested_CPU_time = self.input_trace.loc[waiting_jobs, 'CPU_time'].sum()
        waiting_job_mean_requested_CPU_time = division_helper(waiting_job_total_requested_CPU_time,
                                                              num_waiting_jobs)

        waiting_job_total_requested_wallclock_time = self.input_trace.loc[waiting_jobs, 'requested_time'].sum()
        waiting_job_mean_requested_wallclock_time = division_helper(waiting_job_total_requested_wallclock_time,
                                                                    num_waiting_jobs)

        elapsed_runtime_total = sum(elapsed_runtimes)
        elapsed_runtime_mean = division_helper(elapsed_runtime_total, num_running_jobs)

        elapsed_waiting_time_total = (submit_time - self.input_trace.loc[waiting_jobs, 'submit_time']).sum()
        elapsed_waiting_time_mean = division_helper(elapsed_waiting_time_total, num_waiting_jobs)

        data = np.array([(user_id, submit_time, requested_time, requested_CPUs,
                          num_running_jobs, num_waiting_jobs,
                          running_job_total_requested_CPUs, running_job_total_requested_CPU_time,
                          running_job_mean_requested_CPUs, running_job_mean_requested_CPU_time,
                          running_job_total_requested_wallclock_time, running_job_mean_requested_wallclock_time,
                          waiting_job_total_requested_CPUs, waiting_job_total_requested_CPU_time,
                          waiting_job_mean_requested_CPUs, waiting_job_mean_requested_CPU_time,
                          waiting_job_total_requested_wallclock_time, waiting_job_mean_requested_wallclock_time,
                          elapsed_runtime_total, elapsed_runtime_mean,
                          elapsed_waiting_time_total, elapsed_waiting_time_mean)])

        X_test = pd.DataFrame(data, columns=['user_id', 'submit_time', 'requested_time', 'requested_CPUs',
                                             'num_running_jobs', 'num_waiting_jobs',
                                             'running_job_requested_CPUs', 'running_job_requested_CPU_time',
                                             'running_job_mean_CPUs',
                                             'running_job_mean_CPU_time', 'running_job_requested_wallclock_limit',
                                             'running_job_mean_wallclock_limit',
                                             'waiting_job_requested_CPUs', 'waiting_job_requested_CPU_time',
                                             'waiting_job_mean_CPUs',
                                             'waiting_job_mean_CPU_time', 'waiting_job_requested_wallclock_limit',
                                             'waiting_job_mean_wallclock_limit',
                                             'elapsed_runtime_total', 'elapsed_runtime_mean',
                                             'elapsed_waiting_time_total',
                                             'elapsed_waiting_time_mean'])
        return X_test

    """
        Run the simulator using our custom approach
        *Implements LJW using ML models and a modified form of speculative execution 
        *Implements SWW based on waiting time models
        *First, jobs run on fixed resources if sufficient resources are available
        *Next, any jobs that are predicted to be both long and have a short wait are made to wait for fixed resources
        *All other jobs run on on-demand resources for up to "runtime threshold_min" time. Jobs that are still running
        after this time are killed and resubmitted to queue for fixed resources
    """
    def run_custom_approach(
            self,
            runtime_threshold_min: int,
            waiting_time_threshold_hour: int,
    ) -> None:
        max_wait_time_sec = waiting_time_threshold_hour * SECONDS_IN_HOUR
        runtime_threshold_sec = runtime_threshold_min * 60
        self.input_trace['speculative_execution'] = False

        next_event = self._get_next_event()
        model = WaitingTimeModel(self.training_trace, max_wait_time_sec, self.dataset).train_waiting_time_model()
        total_on_demand_cost = 0

        run_immediately = 0
        run_spec_exec_correct = 0
        run_spec_exec_incorrect = 0
        waited_in_queue = 0
        waited_max_time = 0

        while next_event:
            if next_event.etype == etype.schedule:
                # Schedule new job

                cur_job = self.input_trace.loc[self.input_trace['job_id'] == next_event.job_id].iloc[0]
                cur_job_id, cur_time, cpu_alloc, runtime = get_job_attributes(cur_job)
                is_schedulable = self._can_schedule(cpu_alloc)

                if self.queue_is_empty() and is_schedulable:
                    # Run the job on fixed resources - no queue & sufficient resources available
                    self._schedule(is_schedulable, cur_time, runtime, cur_job_id)
                    self.set_waiting_time(cur_job, 0)
                    run_immediately += 1
                elif is_long_job(cur_job, runtime_threshold_min) and self.is_short_wait(model, cur_job):
                    # Jobs should wait for fixed resources if they are both 1) long and 2) have short waits
                    self.wait_queue.append(cur_job.job_id)
                elif cur_job.speculative_execution:
                    # If the job already ran per speculative execution, it should be made to wait again
                    self.wait_queue.append(cur_job.job_id)
                else:
                    # Run the job on on-demand for up to runtime threshold
                    spec_exec_runtime = min(runtime_threshold_sec, runtime)
                    total_on_demand_cost += self.calc_on_demand_cost_for_job(cur_job, spec_exec_runtime)
                    if runtime <= runtime_threshold_sec:
                        self.set_waiting_time(cur_job, 0)
                        run_spec_exec_correct += 1
                    else:
                        self.resubmit_long_jobs_after_spec_execution(cur_job, runtime_threshold_sec)
                        run_spec_exec_incorrect += 1

            else:
                # A job is finished. So, check the wait queue

                while len(self.wait_queue) > 0:
                    waiting_job = self.input_trace.loc[self.input_trace['job_id'] == self.wait_queue[0]].iloc[0]
                    waiting_job_id, waiting_job_time, waiting_job_cpu_alloc, waiting_job_runtime = \
                        get_job_attributes(waiting_job)
                    time_waited_sec = next_event.time - waiting_job_time + \
                                      (waiting_job.speculative_execution * runtime_threshold_sec)
                    is_waiting_job_schedulable = self._can_schedule(waiting_job_cpu_alloc)

                    # Try scheduling the job
                    if is_waiting_job_schedulable:
                        self.wait_queue.popleft()
                        self._schedule(is_waiting_job_schedulable, next_event.time, waiting_job_runtime, waiting_job_id)
                        self.set_waiting_time(waiting_job, time_waited_sec)
                        waited_in_queue += 1
                    else:
                        break
            next_event = self._get_next_event()

        self.calc_waiting_time_and_reserved_cost()
        self.on_demand_cost = total_on_demand_cost
        self.print_results()
        print("Run immediately: " + str(run_immediately))
        print("Run spec_exec correct: " + str(run_spec_exec_correct))
        print("Run spec_exec incorrect: " + str(run_spec_exec_incorrect))
        print("Waited in queue: " + str(waited_in_queue))
        print("Waited max time: " + str(waited_max_time))

    """
    Run the simulator using original approach vs. which we are benchmarking our custom approach
    *Implements LJW using speculative execution - if sufficient fixed resources are unavailable, all jobs run on 
        on-demand resources for up to "runtime_threshold_min" time before being killed and restarted on fixed resources
        if wait time is predicted to be under "waiting_time_threshold_hour"
    *Implements SWW based on waiting time models - uses ML model to predict whether wait time is under 
        "waiting_time_threshold_hour". Additionally, any jobs that have been waiting longer than this time run on 
        on-demand resources to prevent excessive waits
    """

    def run_original(
            self,
            runtime_threshold_min: int,
            waiting_time_threshold_hour: int,
    ) -> None:
        max_wait_time_sec = waiting_time_threshold_hour * SECONDS_IN_HOUR
        runtime_threshold_sec = runtime_threshold_min * 60

        self.input_trace['speculative_execution'] = False
        total_on_demand_cost = 0

        next_event = self._get_next_event()
        model = WaitingTimeModel(self.training_trace, max_wait_time_sec, self.dataset).train_waiting_time_model()
        run_immediately = 0
        run_spec_exec_correct = 0
        run_on_demand_incorrect = 0
        waited_in_queue = 0
        waited_max_time = 0
        spec_exec_resubmit = 0
        on_demand = 0
        fixed = 0
        num_jobs = 0

        while next_event:
            if next_event.etype == etype.schedule:
                # Schedule new job
                num_jobs += 1
                cur_job = self.input_trace.loc[self.input_trace['job_id'] == next_event.job_id].iloc[0]
                cur_job_id, cur_time, cpu_alloc, runtime = get_job_attributes(cur_job)
                is_schedulable = self._can_schedule(cpu_alloc)

                if self.queue_is_empty() and is_schedulable:
                    # Run the job on fixed resources - no queue & sufficient resources available
                    self.set_waiting_time(cur_job, 0)
                    self._schedule(is_schedulable, cur_time, runtime, cur_job_id)
                    fixed += 1
                    run_immediately += 1
                elif cur_job.speculative_execution:
                    # Jobs should wait for fixed resources if they have short expected waits or already ran under s.e.
                    self.wait_queue.append(cur_job_id)
                else:
                    spec_exec_runtime = min(runtime_threshold_sec, runtime)
                    total_on_demand_cost += self.calc_on_demand_cost_for_job(cur_job, spec_exec_runtime)
                    if runtime <= runtime_threshold_sec:
                        self.set_waiting_time(cur_job, 0)
                        run_spec_exec_correct += 1
                        on_demand += 1
                    else:
                        if self.is_short_wait(model, cur_job):
                            self.resubmit_long_jobs_after_spec_execution(cur_job, runtime_threshold_sec)
                            spec_exec_resubmit += 1
                        else:
                            run_on_demand_incorrect += 1
                            on_demand += 1
                            total_on_demand_cost += self.calc_on_demand_cost_for_job(cur_job,
                                                                                     (runtime - spec_exec_runtime))
                            self.set_waiting_time(cur_job, 0)
            else:
                # A job is finished. So, check the wait queue
                while len(self.wait_queue) > 0:
                    waiting_job = self.input_trace.loc[self.input_trace['job_id'] == self.wait_queue[0]].iloc[0]
                    waiting_job_id, waiting_job_time, waiting_job_cpu_alloc, waiting_job_runtime = \
                        get_job_attributes(waiting_job)
                    time_waited_sec = next_event.time - waiting_job_time + \
                                      (waiting_job.speculative_execution * runtime_threshold_sec)
                    is_waiting_job_schedulable = self._can_schedule(waiting_job_cpu_alloc)

                    # Check if job waited more than the max waiting time
                    if time_waited_sec > (max_wait_time_sec):
                        # Remove the job from the queue
                        self.wait_queue.popleft()
                        total_on_demand_cost += self.calc_on_demand_cost_for_job(waiting_job, waiting_job_runtime)
                        self.set_waiting_time(waiting_job, max_wait_time_sec)
                        waited_max_time += 1
                        on_demand += 1
                    else:
                        # Try scheduling the job
                        if is_waiting_job_schedulable:
                            self.wait_queue.popleft()
                            self._schedule(is_waiting_job_schedulable, next_event.time,
                                           waiting_job_runtime, waiting_job_id)
                            self.set_waiting_time(waiting_job, time_waited_sec)
                            waited_in_queue += 1
                            fixed += 1
                        else:
                            break
            next_event = self._get_next_event()

        self.calc_waiting_time_and_reserved_cost()
        self.on_demand_cost = total_on_demand_cost
        self.print_results()
        print("Run immediately: " + str(run_immediately))
        print("Run spec_exec correct: " + str(run_spec_exec_correct))
        print("Run spec_exec incorrect: " + str(run_on_demand_incorrect))
        print("Waited in queue: " + str(waited_in_queue))
        print("Waited max time: " + str(waited_max_time))
        print("Resubmitted spec_exec: " + str(spec_exec_resubmit))
        print("On-demand: " + str(on_demand))
        print("Fixed: " + str(fixed))
        print("Jobs remaining in queue: " + str(len(self.wait_queue)))
        print("Num jobs:" + str(num_jobs))
    """
    Run the simulator with NJW policy - no jobs wait for fixed resources; all jobs run on-demand as needed
    """

    def run_NJW(self) -> None:
        total_on_demand_cost = 0
        prev_time = self.input_trace.iloc[0].submit_time

        while self.next_jobID < self.job_count:
            # New job request
            cur_job = self.input_trace.loc[self.input_trace['job_id'] == self.next_jobID].iloc[0]
            cur_job_id, cur_time, cpu_alloc, runtime = get_job_attributes(cur_job)
            is_schedulable = self._can_schedule(cpu_alloc)

            if cur_time != prev_time:
                self._check_expired_jobs(cur_time)

            if is_schedulable:
                # Run the job on fixed resources - no wait queue
                self._schedule(is_schedulable, cur_time, runtime, cur_job_id)
                self.set_waiting_time(cur_job, 0)
            else:
                # Job should run on on-demand resources instead of waiting
                self.set_waiting_time(cur_job, 0)
                total_on_demand_cost += self.calc_on_demand_cost_for_job(cur_job, runtime)
            self.next_jobID += 1
            prev_time = cur_time

        self.calc_waiting_time_and_reserved_cost()
        self.on_demand_cost = total_on_demand_cost
        self.print_results()

    """
    Run the simulator with AJW policy - no jobs run on on-demand resources; all jobs wait for fixed resources
    """

    def run_AJW(
            self
    ) -> None:

        next_event = self._get_next_event()

        while next_event:
            if next_event.etype == etype.schedule:
                # New job request
                cur_job = self.input_trace.loc[self.input_trace['job_id'] == next_event.job_id].iloc[0]

                cur_job_id, cur_time, cpu_alloc, runtime = get_job_attributes(cur_job)
                is_schedulable = self._can_schedule(cpu_alloc)

                if self.queue_is_empty() and is_schedulable:
                    # Run the job on fixed resources - no wait queue
                    self._schedule(is_schedulable, cur_time, runtime, cur_job_id)
                    self.set_waiting_time(cur_job, 0)
                else:
                    # Job should run on fixed resources, but it must wait
                    self.wait_queue.append(cur_job.job_id)
            else:
                # A job is finished. So, check the wait queue
                while len(self.wait_queue) > 0:

                    waiting_job = self.input_trace.iloc[self.wait_queue[0]]
                    waiting_job_id, waiting_job_time, waiting_job_cpu_alloc, waiting_job_runtime = \
                        get_job_attributes(waiting_job)

                    time_waited_sec = next_event.time - waiting_job_time
                    is_waiting_job_schedulable = self._can_schedule(waiting_job_cpu_alloc)

                    if is_waiting_job_schedulable:
                        self._schedule(is_waiting_job_schedulable, next_event.time, waiting_job_runtime, waiting_job_id)
                        self.wait_queue.popleft()
                        self.set_waiting_time(waiting_job, time_waited_sec)

                    else:
                        break
            next_event = self._get_next_event()

        self.calc_waiting_time_and_reserved_cost()
        self.on_demand_cost = 0
        self.print_results()


def get_results_custom_approach(simulator, dataset, runtime_threshold, waiting_time_threshold) -> None:
    print("Running custom approach on " + dataset + " with runtime threshold of " + str(runtime_threshold) +
          " minutes and waiting time threshold of " + str(waiting_time_threshold) + " hour(s)")
    simulator.run_custom_approach_with_waiting_time_limit(runtime_threshold_min=runtime_threshold,
                                  waiting_time_threshold_hour=waiting_time_threshold)


def get_results_original(simulator, dataset, runtime_threshold, waiting_time_threshold) -> None:
    print("Running original approach on " + dataset + " with runtime threshold of " + str(runtime_threshold) +
          " minutes and waiting time threshold of " + str(waiting_time_threshold) + " hour(s)")
    simulator.run_original(runtime_threshold_min=runtime_threshold, waiting_time_threshold_hour=waiting_time_threshold)

def get_results_NJW(simulator, dataset) -> None:
    print("Running NJW on " + dataset)
    simulator.run_NJW()

def get_results_AJW(simulator, dataset) -> None:
    print("Running AJW on " + dataset)
    simulator.run_AJW()

if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset == 'ANL':
        input_trace = pd.read_csv(os.path.abspath('../Dataset/ANL_trace_output_MODIFIED_LATEST_bayes.csv')) \
            .sort_values(by=['submit_time'])
        training_trace = pd.read_csv(os.path.abspath('../Dataset/cleaned_ANL_with_waiting_times_full.csv'))
        num_VMs = 40960
        vm_CPU = 4
    elif dataset == 'RICC':
        input_trace = pd.read_csv(os.path.abspath('../Dataset/RICC_trace_output_MODIFIED.csv')) \
            .sort_values(by=['submit_time'])
        training_trace = pd.read_csv(os.path.abspath('../Dataset/cleaned_RICC_with_waiting_times_full.csv'))
        num_VMs = 1024
        vm_CPU = 8
    else:
        exit(1)

    simulator = Simulator(dataset=dataset, NUM_VMS=num_VMs, VM_CPU=vm_CPU, input_trace=input_trace, training_trace=training_trace)
    for runtime_threshold in RUNTIME_THRESHOLD_VALUES:
        for waiting_time_threshold in WAITING_TIME_THRESHOLD_VALUES:
           simulator = Simulator(dataset=dataset, NUM_VMS=num_VMs, VM_CPU=vm_CPU, input_trace=input_trace,
                                  training_trace=training_trace)
           get_results_original(simulator, dataset, runtime_threshold, waiting_time_threshold)