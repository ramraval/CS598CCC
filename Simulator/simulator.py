"""
Copyright (c) 2020 <Sustainable Computing Lab>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import sys
import time

from collections import deque
from enum import Enum
from typing import Dict, List, Optional

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sortedcontainers import SortedList
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

from node import Node

UNIT_CPU_COST_HR = 0.048  # Cost Estimation assuming m5 family
RESERVE_DISCOUNT = 0.6  # 3 year discount
SECONDS_IN_HOUR = 3600
TEST_SIZE = 0.1
RUNTIME_THRESHOLD_VALUES = [5, 10, 15, 30, 60]
WAITING_TIME_THRESHOLD_VALUES = [1, 3, 6, 12, 24]

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
Class definition for simulator

Instance Variables:
NUM_VMS: int; Number of VMs/nodes in cluster
VM_CPU: int; number of CPU cores per VM
input_trace: Dataframe; Input trace as pandas data frame

next_jobID: int; jobID of next job arrival
current_reserve_jobs: SortedList(tuple); tuple - (finishtime, jobID, nodeID)
job_schedule_map: Dictionary; key - jobID, value - 'R' or 'D'
wait_time_map: Dictionary; key - jobID, value - int
wait_queue: deque; waiting queue
"""


def division_helper(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0


def get_job_attributes(cur_job):
    cur_job_id, cur_time, cpu_alloc, runtime = cur_job.job_id, cur_job.submit_time, cur_job.allocated_CPUs, \
                                               cur_job.runtime
    return cur_job_id, cur_time, cpu_alloc, runtime


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

        # Variables
        self.next_jobID: int = 0
        self.current_reserve_jobs: SortedList = SortedList()
        self.job_schedule_map: Dict[int, str] = {}
        self.wait_time_map: Dict[int, int] = {}
        self.wait_queue: deque = deque()

        self.mean_waiting_time: float
        self.reserved_cost: float
        self.on_demand_Cost: float

        # Cluster
        self._reserve_nodes: List[Node] = []
        vm_name = "m5.16xlarge"
        for i in range(NUM_VMS):
            n = Node(vm_name, i, VM_CPU)
            self._reserve_nodes.append(n)

    def _first_fit_scheduling(self, cpu_req: int) -> Optional[dict]:
        counter = 0
        node_dic = {}

        # jobs need to be scheduled across multiple nodes
        while cpu_req and counter < len(self._reserve_nodes):
            node = self._reserve_nodes[counter]
            cpus_scheduled = min(cpu_req, node.idle_cpu)
            if cpus_scheduled:
                node_dic[node._nodeID] = cpus_scheduled
            cpu_req -= cpus_scheduled
            counter += 1

        if cpu_req == 0:
            return node_dic
        else:
            return None

    """
      VM_CPU: number of CPUs per VM on fixed resources (will mimic for on-demand resources)
      ondemand_df: input ondemand jobs dataframe
      """

    def _compute_ondemand_cost(self) -> float:
        ondemand_df = self.input_trace[self.input_trace["schedule"] == "D"]
        ondemand_df['num_nodes'] = self.VM_CPU * round(ondemand_df['allocated_CPUs'] / self.VM_CPU)
        ondemand_df['cost'] = ondemand_df['num_nodes'] * self.VM_CPU * UNIT_CPU_COST_HR * \
                              (ondemand_df['runtime'] / SECONDS_IN_HOUR)
        return ondemand_df.cost.sum()

    def _compute_fixed_cost(self) -> float:
        start_time = self.input_trace.loc[self.input_trace['submit_time'].idxmin()].submit_time
        end_time = self.input_trace.loc[self.input_trace['end_time'].idxmax()].end_time
        hours_in_operation = (end_time - start_time) / SECONDS_IN_HOUR
        return hours_in_operation * UNIT_CPU_COST_HR * self.NUM_VMS * self.VM_CPU * (1 - RESERVE_DISCOUNT)

    def is_long_job(self, cur_job, runtime_threshold) -> bool:
        return cur_job['should_wait_runtime_predicted_' + str(runtime_threshold)]

    """
  Schedule the job using first-fit online packing algorithm
  """

    def _can_schedule(
            self,
            start_time: int,
            cpu_req: int,
            runtime_sec: int,
            job_id: int,
    ) -> bool:
        node_dic = self._first_fit_scheduling(cpu_req)
        if node_dic:
            for node_id, CPU_count in node_dic.items():
                self._reserve_nodes[node_id].schedule_job(CPU_count, job_id)
                self._reserve_nodes[node_id].update_cpu_time_utilized(CPU_count, runtime_sec)
            finish_time = start_time + runtime_sec
            self.current_reserve_jobs.add((finish_time, job_id, node_dic.keys()))
            return True
        else:
            return False

    """
    Check for the finished jobs on reserved nodes, given the current time.
    Add the back the resources to the respective nodes.
    """

    def _check_expired_jobs(self, current_time: int) -> None:
        while self.current_reserve_jobs:
            finish_time, job_id, node_ids = self.current_reserve_jobs[0]
            if finish_time <= current_time:
                for node_id in node_ids:
                    node = self._reserve_nodes[node_id]
                    # Remove the job from the node
                    node.remove_job(job_id)
                    # Pop the job from the running job list
                self.current_reserve_jobs.pop(0)
            else:
                break

    def train_waiting_time_model(self, max_wait_time_sec: int):
        df = self.training_trace
        df['should_wait_waiting_time_actual'] = df['wait_time'] < max_wait_time_sec

        X = df[['user_id', 'group_id', 'submit_time', 'requested_time', 'requested_CPUs', 'requested_memory',
                'num_running_jobs', 'num_waiting_jobs',
                'running_job_requested_CPUs', 'running_job_requested_CPU_time', 'running_job_mean_CPUs',
                'running_job_mean_CPU_time', 'running_job_requested_wallclock_limit',
                'running_job_mean_wallclock_limit',
                'waiting_job_requested_CPUs', 'waiting_job_requested_CPU_time', 'waiting_job_mean_CPUs',
                'waiting_job_mean_CPU_time', 'waiting_job_requested_wallclock_limit',
                'waiting_job_mean_wallclock_limit',
                'elapsed_runtime_total', 'elapsed_runtime_mean', 'elapsed_waiting_time_total',
                'elapsed_waiting_time_mean']]
        y = df['should_wait_waiting_time_actual']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

        if self.dataset == 'ANL':
            clf = GradientBoostingClassifier(n_estimators=100, random_state=0)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=0)

        clf.fit(X_train, y_train.values.ravel())
        return clf

    def calc_system_state_features(self, waiting_job, current_reserve_jobs: SortedList,
                                   wait_queue: deque) -> pd.DataFrame:
        running_job_ids = [job[1] for job in current_reserve_jobs]
        running_jobs = self.input_trace['job_id'].isin(running_job_ids)
        waiting_jobs = self.input_trace['job_id'].isin(wait_queue)

        user_id = waiting_job.user_id
        group_id = waiting_job.group_id
        submit_time = waiting_job.submit_time
        requested_time = waiting_job.requested_time
        requested_CPUs = waiting_job.requested_CPUs
        requested_memory = waiting_job.requested_memory

        num_running_jobs = len(current_reserve_jobs)
        num_waiting_jobs = len(wait_queue)

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

        elapsed_runtime_total = (submit_time - self.input_trace.loc[running_jobs, 'start_time']).sum()
        elapsed_runtime_mean = division_helper(elapsed_runtime_total, num_running_jobs)

        elapsed_waiting_time_total = (submit_time - self.input_trace.loc[waiting_jobs, 'submit_time']).sum()
        elapsed_waiting_time_mean = division_helper(elapsed_waiting_time_total, num_waiting_jobs)

        data = np.array([(user_id, group_id, submit_time, requested_time, requested_CPUs, requested_memory,
                          num_running_jobs, num_waiting_jobs,
                          running_job_total_requested_CPUs, running_job_total_requested_CPU_time,
                          running_job_mean_requested_CPUs, running_job_mean_requested_CPU_time,
                          running_job_total_requested_wallclock_time, running_job_mean_requested_wallclock_time,
                          waiting_job_total_requested_CPUs, waiting_job_total_requested_CPU_time,
                          waiting_job_mean_requested_CPUs, waiting_job_mean_requested_CPU_time,
                          waiting_job_total_requested_wallclock_time, waiting_job_mean_requested_wallclock_time,
                          elapsed_runtime_total, elapsed_runtime_mean,
                          elapsed_waiting_time_total, elapsed_waiting_time_mean)])

        X_test = pd.DataFrame(data, columns=['user_id', 'group_id', 'submit_time', 'requested_time',
                                             'requested_CPUs', 'requested_memory',
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
  Returns the next event
  """

    def _get_next_event(self) -> Optional[Event]:
        next_arr: pd.Series = None
        next_dep: pd.Series = None
        next_arr_time = float('inf')
        next_dep_time = float('inf')

        # Next Arrival
        if self.next_jobID < self.job_count:
            next_arr = self.input_trace.iloc[self.next_jobID]
            next_arr_time = next_arr.submit_time

        # Next Departure
        if self.current_reserve_jobs:
            next_dep_time, dep_job_id, _ = self.current_reserve_jobs[0]
            next_dep = self.input_trace.iloc[dep_job_id]

        # All the events are played (0, 0)
        if next_arr is None and next_dep is None:
            return None

        # Compare time events
        if next_dep_time <= next_arr_time:
            next_event = Event(etype.finish, next_dep_time, next_dep.job_id)
            # Add back the capacity
            _, _, node_ids = self.current_reserve_jobs.pop(0)
            for node_id in node_ids:
                self._reserve_nodes[node_id].remove_job(next_dep.job_id)
            return next_event
        else:
            next_event = Event(etype.schedule, next_arr_time, next_arr.job_id)
            self.next_jobID += 1
            return next_event

    def is_short_wait(self, model, cur_job) -> bool:
        X_test = self.calc_system_state_features(cur_job, self.current_reserve_jobs, self.wait_queue)
        return model.predict(X_test)

    def queue_is_empty(self):
        return not self.wait_queue

    def print_results(self) -> None:
        print(
            "Reserved cost is ",
            self.reserved_cost,
            ", On demand cost is ",
            self.on_demand_cost,
            ", Total cost is ",
            self.reserved_cost + self.on_demand_cost,
            ", Mean waiting time is ",
            self.mean_waiting_time
        )

    """
    Run the simulator (event based) with waiting queue. Applicable to LJW and compound policy.
    """

    def run_custom_approach(
            self,
            runtime_threshold_min: int,
            waiting_time_threshold_hour: int,
    ) -> None:
        max_wait_time_sec = waiting_time_threshold_hour * SECONDS_IN_HOUR
        next_event = self._get_next_event()
        model = self.train_waiting_time_model(max_wait_time_sec)

        while next_event:
            if next_event.etype == etype.schedule:
                # Schedule new job

                cur_job = self.input_trace.iloc[next_event.job_id]
                cur_job_id, cur_time, cpu_alloc, runtime = get_job_attributes(cur_job)

                if self.queue_is_empty() and self._can_schedule(cur_time, cpu_alloc, runtime,
                                                                next_event.job_id):
                    # Run the job on fixed resources - no queue & sufficient resources available
                    self.mark_as_fixed_resource_job(cur_job, 0)
                elif self.is_long_job(cur_job, runtime_threshold_min) or self.is_short_wait(model, cur_job):
                    # Jobs should wait for fixed resources if they are 1) long or 2) have short waits
                    self.wait_queue.append(cur_job.job_id)
                else:
                    # Run the job on on-demand -- short job or long wait
                    self.mark_as_on_demand_resource_job(cur_job, 0)

            else:
                # A job is finished. So, check the wait queue
                while len(self.wait_queue) > 0:
                    waiting_job = self.input_trace.iloc[self.wait_queue[0]]
                    waiting_job_id, waiting_job_time, waiting_job_cpu_alloc, waiting_job_runtime = \
                        get_job_attributes(waiting_job)
                    time_waited_sec = next_event.time - waiting_job_time

                    # Check if job waited more than the max waiting time
                    if time_waited_sec > max_wait_time_sec:
                        # Remove the job from the queue
                        self.wait_queue.popleft()
                        self.mark_as_on_demand_resource_job(waiting_job, max_wait_time_sec)
                    else:
                        # Try scheduling the job
                        if self._can_schedule(
                                next_event.time, waiting_job_cpu_alloc, waiting_job_runtime, waiting_job_id
                        ):
                            self.wait_queue.popleft()
                            self.mark_as_fixed_resource_job(waiting_job, time_waited_sec)
                        else:
                            break
            next_event = self._get_next_event()

        self.calc_simulator_attributes()
        self.print_results()

    def calc_simulator_attributes(self):
        self.input_trace["schedule"] = (
            self.input_trace["job_id"].map(self.job_schedule_map)
        )
        self.input_trace["wait_time_sec"] = (
            self.input_trace["job_id"].map(self.wait_time_map)
        )
        self.mean_waiting_time, self.on_demand_cost, self.reserved_cost = self.input_trace.wait_time_sec.mean(), \
                                                                          self._compute_ondemand_cost(), self._compute_fixed_cost()

    def mark_as_on_demand_resource_job(self, cur_job, waiting_time):
        self.job_schedule_map[cur_job.job_id] = "D"
        self.wait_time_map[cur_job.job_id] = waiting_time

    def mark_as_fixed_resource_job(self, cur_job, waiting_time):
        self.job_schedule_map[cur_job.job_id] = "R"
        self.wait_time_map[cur_job.job_id] = waiting_time

    """
        Run the simulator (event based) with waiting queue. Applicable to LJW and compound policy.
        """

    def run_Ambati(
            self,
            sww_model: GradientBoostingClassifier,
            max_wait_time_min: int,
            short_thresh_min: int,
    ) -> None:
        # start time of simulation; for measuring the total simulation time
        t_start = time.time()

        # book keeping
        reserve_cpu_time = 0
        on_demand_count = 0

        total_on_demand_cost = 0
        reserve_jobs_run = 0
        y_pred_count = 0
        y_pred_total = 0

        # Next event for the simulator
        next_event = self._get_next_event()

        # For computing cost
        start = end = self.input_trace.iloc[0].submit_time

        self.input_trace['speculative_execution'] = False

        while next_event:
            if next_event.etype == etype.schedule:
                # New job request
                cur_job = self.input_trace.iloc[next_event.job_id]
                cur_time = next_event.time

                # Job requirements
                cpu_req = cur_job.allocated_CPUs
                runtime_sec = cur_job.runtime

                if (not self.wait_queue) and self._can_schedule(
                        cur_time, cpu_req, runtime_sec, next_event.job_id):
                    # Run the job on fixed resources - no wait queue
                    reserve_cpu_time += cpu_req * runtime_sec
                    end = max(end, cur_time + runtime_sec)
                    self.mark_as_fixed_resource_job(cur_job)
                    reserve_jobs_run += 1
                elif cur_job.speculative_execution:
                    self.wait_queue.append(cur_job.job_id)
                else:
                    on_demand_nodes_needed = self.VM_CPU * round(cur_job.allocated_CPUs / self.VM_CPU)
                    spec_exec_runtime = min(short_thresh_min * 60, cur_job.runtime)
                    total_on_demand_cost += (
                            on_demand_nodes_needed * self.VM_CPU * (spec_exec_runtime / SECONDS_IN_HOUR)
                            * UNIT_CPU_COST_HR)
                    if cur_job.runtime <= (short_thresh_min * 60):
                        self.wait_time_map[cur_job.job_id] = 0
                        end = max(end, cur_time + runtime_sec)
                    else:
                        X_test = self.calc_system_state_features(cur_job, self.current_reserve_jobs, self.wait_queue)
                        y_pred = sww_model.predict(X_test)
                        y_pred_total += 1
                        if not y_pred:
                            y_pred_count += 1
                            remaining_runtime = cur_job.runtime - (short_thresh_min * 60)
                            total_on_demand_cost += (on_demand_nodes_needed * self.VM_CPU * (
                                    remaining_runtime / SECONDS_IN_HOUR)
                                                     * UNIT_CPU_COST_HR)

                            self.wait_time_map[cur_job.job_id] = 0
                            end = max(end, cur_time + runtime_sec)
                        else:
                            cur_job.submit_time = cur_job.submit_time + (short_thresh_min * 60)
                            cur_job.speculative_execution = True
                            temp_df = self.input_trace[self.input_trace['submit_time'] > cur_job.submit_time]
                            new_index = temp_df.iloc[0].job_id
                            cur_job.job_id = new_index
                            self.input_trace = pd.concat([self.input_trace.iloc[:new_index], cur_job,
                                                          self.input_trace.iloc[new_index:]]).reset_index(drop=True)
                            self.input_trace['job_id'] = self.input_trace.index
            else:
                # A job is finished. So, check the wait queue
                while len(self.wait_queue) > 0:
                    job_id = self.wait_queue[0]
                    waiting_job = self.input_trace.iloc[job_id]
                    time_waited_sec = next_event.time - waiting_job.submit_time + (short_thresh_min * 60)

                    # Compute maximum balking time
                    max_wait_time_sec = max_wait_time_min * 60

                    # Check if job waited more than the balking time
                    if time_waited_sec > max_wait_time_sec:
                        # Remove the job from the queue
                        self.wait_queue.popleft()
                        self.wait_time_map[job_id] = max_wait_time_sec
                        on_demand_count += 1
                        on_demand_nodes_needed = self.VM_CPU * round(waiting_job.allocated_CPUs / self.VM_CPU)
                        total_on_demand_cost += (on_demand_nodes_needed * self.VM_CPU *
                                                 (waiting_job.runtime / SECONDS_IN_HOUR) * UNIT_CPU_COST_HR)

                        end = max(end, next_event.time + waiting_job.runtime)

                    else:
                        # Try scheduling the job
                        cpu_req = waiting_job.allocated_CPUs
                        runtime_sec = waiting_job.runtime
                        if self._can_schedule(
                                next_event.time, cpu_req, runtime_sec, job_id
                        ):
                            # Book keeping
                            reserve_cpu_time += cpu_req * runtime_sec

                            # remove the job from the queue
                            self.wait_queue.popleft()

                            # Add waiting time to wait times map
                            self.wait_time_map[job_id] = time_waited_sec
                            self.job_schedule_map[job_id] = "R"

                            end = max(end, next_event.time + runtime_sec)
                        else:
                            break
            next_event = self._get_next_event()

        t_end = time.time()
        print("Time taken in seconds is ", t_end - t_start)

        self.input_trace["wait_time_sec"] = (
            self.input_trace["job_id"].map(self.wait_time_map)
        )

        mean_wait_time_sec = self.input_trace.wait_time_sec.mean()

        # Simulation time
        elapsed_simulation_time = (end - start)
        duration_hrs = elapsed_simulation_time / 3600

        # Computing total cost
        reserved_cost = 2000000

        print(
            "Reserved cost is ",
            reserved_cost,
            ", On demand cost is ",
            total_on_demand_cost,
            ", Total cost is ",
            reserved_cost + total_on_demand_cost,
            ", Mean waiting time is ",
            mean_wait_time_sec,
            "Reserve jobs run ",
            reserve_jobs_run,
            "y_pred_count ",
            y_pred_count / y_pred_total
        )
        print("----------------------------------------------------------------")

    """
       Run the simulator (event based) with waiting queue. Applicable to LJW and compound policy.
       """

    def run_NJW(
            self,
    ) -> None:
        # start time of simulation; for measuring the total simulation time
        t_start = time.time()

        # book keeping
        reserve_cpu_time = 0
        on_demand_count = 0

        cur_time = prev_time = self.input_trace.iloc[0].submit_time

        # For computing cost
        start = end = self.input_trace.iloc[0].submit_time

        while self.next_jobID < self.job_count:
            # New job request
            next_job = self.input_trace.iloc[self.next_jobID]
            cur_time = next_job.submit_time

            if cur_time != prev_time:
                self._check_expired_jobs(cur_time)

            # Job requirements
            cpu_req = next_job.allocated_CPUs
            runtime_sec = next_job.runtime

            end = max(end, cur_time + runtime_sec)

            if self._can_schedule(
                    cur_time, cpu_req, runtime_sec, next_job.job_id):
                # Run the job on fixed resources - no wait queue
                reserve_cpu_time += cpu_req * runtime_sec
                self.mark_as_fixed_resource_job(next_job)
            else:
                # Job should run on on-demand resources instead of waiting
                on_demand_count += 1
                self.mark_as_on_demand_resource_job(next_job)
            self.next_jobID += 1
            prev_time = cur_time

        t_end = time.time()
        print("Time taken in seconds is ", t_end - t_start)

        # Post- Simulation: Add schedule column and wait_time column to input dataframe
        self.input_trace["schedule"] = (
            self.input_trace["job_id"].map(self.job_schedule_map)
        )
        self.input_trace["wait_time_sec"] = (
            self.input_trace["job_id"].map(self.wait_time_map)
        )

        mean_wait_time_sec = self.input_trace.wait_time_sec.mean()

        # Simulation time
        elapsed_simulation_time = (end - start)
        duration_hrs = elapsed_simulation_time / 3600

        # Computing total cost
        (
            total_cost,
            reserved_cost,
            on_demand_cost,
        ) = self._compute_total_cost(duration_hrs)
        print(
            "Reserved cost is ",
            reserved_cost,
            ", On demand cost is ",
            on_demand_cost,
            ", Total cost is ",
            reserved_cost + on_demand_cost,
            ", Mean waiting time is ",
            mean_wait_time_sec,
        )
        print("----------------------------------------------------------------")

    """
    Run the simulator (event based) with waiting queue. Applicable to LJW and compound policy.
    """

    def run_AJW(
            self
    ) -> None:
        # start time of simulation; for measuring the total simulation time
        t_start = time.time()

        # book keeping
        reserve_cpu_time = 0
        on_demand_count = 0

        # Next event for the simulator
        next_event = self._get_next_event()

        # For computing cost
        start = end = self.input_trace.iloc[0].submit_time

        while next_event:
            if next_event.etype == etype.schedule:
                # New job request
                cur_job = self.input_trace.iloc[next_event.job_id]
                cur_time = next_event.time

                # Job requirements
                cpu_req = cur_job.allocated_CPUs
                runtime_sec = cur_job.runtime

                if (not self.wait_queue) and self._can_schedule(
                        cur_time, cpu_req, runtime_sec, next_event.job_id):
                    # Run the job on fixed resources - no wait queue
                    reserve_cpu_time += cpu_req * runtime_sec
                    end = max(end, cur_time + runtime_sec)
                    self.mark_as_fixed_resource_job(cur_job)
                else:
                    # Job should run on fixed resources, but it must wait
                    self.wait_queue.append(cur_job.job_id)

            else:
                # A job is finished. So, check the wait queue
                while len(self.wait_queue) > 0:
                    job_id = self.wait_queue[0]
                    waiting_job = self.input_trace.iloc[job_id]
                    time_waited_sec = next_event.time - waiting_job.submit_time

                    # Try scheduling the job
                    cpu_req = waiting_job.allocated_CPUs
                    runtime_sec = waiting_job.runtime
                    if self._can_schedule(
                            next_event.time, cpu_req, runtime_sec, job_id
                    ):
                        # Book keeping
                        reserve_cpu_time += cpu_req * runtime_sec

                        # remove the job from the queue
                        self.wait_queue.popleft()

                        # Add waiting time to wait times map
                        self.wait_time_map[job_id] = time_waited_sec
                        self.job_schedule_map[job_id] = "R"

                        end = max(end, next_event.time + runtime_sec)
                    else:
                        break
            next_event = self._get_next_event()

        t_end = time.time()
        print("Time taken in seconds is ", t_end - t_start)

        # Post- Simulation: Add schedule column and wait_time column to input dataframe
        self.input_trace["schedule"] = (
            self.input_trace["job_id"].map(self.job_schedule_map)
        )
        self.input_trace["wait_time_sec"] = (
            self.input_trace["job_id"].map(self.wait_time_map)
        )

        mean_wait_time_sec = self.input_trace.wait_time_sec.mean()

        # Simulation time
        elapsed_simulation_time = (end - start)
        duration_hrs = elapsed_simulation_time / 3600

        # Computing total cost
        (
            total_cost,
            reserved_cost,
            on_demand_cost,
        ) = self._compute_total_cost(duration_hrs)
        print(
            "Reserved cost is ",
            reserved_cost,
            ", On demand cost is ",
            on_demand_cost,
            ", Total cost is ",
            reserved_cost + on_demand_cost,
            ", Mean waiting time is ",
            mean_wait_time_sec,
        )
        print("----------------------------------------------------------------")


def get_results_custom_approach(simulator, dataset, runtime_threshold, waiting_time_threshold) -> None:
    print("Running custom approach on " + dataset + " with runtime threshold of " + str(runtime_threshold) +
          " minutes and waiting time threshold of " + str(waiting_time_threshold) + " hour(s)")
    simulator.run_custom_approach(runtime_threshold_min=runtime_threshold,
                                  waiting_time_threshold_hour=waiting_time_threshold)


if __name__ == "__main__":
    dataset = sys.argv[1]
    if dataset == 'ANL':
        input_trace = pd.read_csv(os.path.abspath('../Dataset/ANL_trace_output_MODIFIED.csv')) \
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

    for runtime_threshold in RUNTIME_THRESHOLD_VALUES:
        for waiting_time_threshold in WAITING_TIME_THRESHOLD_VALUES:
            simulator = Simulator(dataset=dataset, NUM_VMS=num_VMs, VM_CPU=vm_CPU, input_trace=input_trace,
                                  training_trace=training_trace)
            get_results_custom_approach(simulator, dataset, runtime_threshold, waiting_time_threshold)
