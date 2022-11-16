"""
Copyright (c) 2020 <Sustainable Computing Lab>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import time
import pytz

from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
from sortedcontainers import SortedList
import pandas as pd

from node import Node

# Cost Estimation assuming m5 family
INSTANCE_CPUS = [0, 1, 2, 4, 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
UNIT_CPU_COST_HR = 0.048
RESERVE_DISCOUNT = 0.6  # 3 year discount


def estimate_ondemand_cost(ondemand_df: pd.DataFrame) -> float:
  """
  ondemand_df: input ondemand jobs dataframe
  """

  best_cost = 0.0
  i = 0
  for j in range(i, len(INSTANCE_CPUS) - 1):
    curr_cost = compute_perfect_fit(
      INSTANCE_CPUS[j], INSTANCE_CPUS[j + 1], ondemand_df
    )
    best_cost += curr_cost

  return best_cost


def compute_perfect_fit(b1: int, b2: int, df: pd.DataFrame) -> float:
  """
  type b1: lowerbound on cpu
  type b2: upperbound on cpu
  type df: dataframe
  """

  # Todo: Figure out what to do here
  #filtered_df = df[(df["cpu_dominant"] > b1) & (df["cpu_dominant"] <= b2)]
  #bounded_job_rt = filtered_df.wallclock_runtime_sec.sum() / 3600

  bounded_job_rt = df.wallclock_runtime_sec.sum() / 3600
  return b2 * bounded_job_rt * 0.048


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
  def __init__(self, etype: etype, time: datetime, job_id: int):
    self.etype = etype
    self.time = time
    self.job_id = job_id


"""
Class definition for simulator

Instance Variables:
_year: int
NUM_VMS: int; Number of VMs/nodes in cluster
VM_CPU: int; number of CPU cores per VM
input_trace: Dataframe; Input trace as pandas data frame
next_jobID: int; jobID of next job arrival
current_reserve_jobs: SortedList(tuple); tuple - (finishtime, jobID, nodeID)
job_schedule_map: Dictionary; key - jobID, value - 'R' or 'D'
wait_time_map: Dicitionary; key - jobID, value - int
wait_queue: deque; waiting queue
"""


class Simulator:
  def __init__(
          self,
          NUM_VMS: int,
          VM_CPU: int,
          input_trace: pd.DataFrame,
  ) -> None:
    self.NUM_VMS = NUM_VMS
    self.VM_CPU = VM_CPU

    self.input_trace = input_trace
    self.job_count = input_trace.time.count()

    # Variables
    self.next_jobID: int = 0
    self.current_reserve_jobs: SortedList = SortedList()
    self.job_schedule_map: Dict[int, str] = {}
    self.wait_time_map: Dict[int, int] = {}
    self.wait_queue: deque = deque()

    # Cluster
    self._reserve_nodes: List[Node] = []
    vm_name = "m5.16xlarge"
    for i in range(NUM_VMS):
      n = Node(vm_name, i, VM_CPU)
      self._reserve_nodes.append(n)

  """
  First fit packing
  """

  def _first_fit_scheduling(self, cpu_req: int) -> Optional[Node]:
    # Return the first node that can schedule
    for node in self._reserve_nodes:
      if node.can_schedule(cpu_req):
        return node
    return None

  """
  Schedule the job using preferred online packing algorithm
  """

  def _schedule_job(
          self,
          start_time: datetime,
          cpu_req: int,
          runtime_sec: int,
          job_id: int,
  ) -> bool:
    node = self._first_fit_scheduling(cpu_req)
    if node:
      node.schedule_job(cpu_req, job_id)
      node.update_cpu_time_utilized(cpu_req, runtime_sec)
      finish_time = start_time + timedelta(seconds=int(runtime_sec))
      self.current_reserve_jobs.add((finish_time, job_id, node._nodeID))
      self.job_schedule_map[job_id] = "R"
      return True
    else:
      return False

  """
  Given the scheduling of the jobs and total duration in hours, compute the total cost.
  Output is total cost, reserved cost and on_demand cost
  """

  def _compute_total_cost(
          self, input_trace: pd.DataFrame, duration_hrs: float
  ) -> tuple:
    reserved_cost = (
            self.NUM_VMS
            * self.VM_CPU
            * UNIT_CPU_COST_HR
            * (1 - RESERVE_DISCOUNT)
            * duration_hrs
    )
    ondemand_df = self.input_trace[self.input_trace["schedule"] == "D"]
    on_demand_cost = estimate_ondemand_cost(ondemand_df)
    total_cost = reserved_cost + on_demand_cost

    # Computing amortized cost
    mean_service_time_hr = self.input_trace.wallclock_runtime_sec.mean() / 3600
    mean_job_req = self.input_trace.num_cores.mean()
    normalized_price = total_cost / (
            mean_service_time_hr * self.job_count * mean_job_req * UNIT_CPU_COST_HR
    )
    print("Normalized price is ", normalized_price)

    return (total_cost, reserved_cost, on_demand_cost, normalized_price)

  """
  Method compares next job start time against current_reserve_job, and returns the job
  with the lower timestamp.
  """

  def _get_next_event(self) -> Optional[Event]:
    next_arr: pd.Series = None
    next_dep: pd.Series = None
    next_arr_time: datetime = datetime.max
    next_dep_time: datetime = datetime.max

    next_arr_time = pytz.utc.localize(next_arr_time)
    next_dep_time = pytz.utc.localize(next_dep_time)

    # Next Arrival
    if self.next_jobID < self.job_count:
      next_arr = self.input_trace.iloc[self.next_jobID]
      next_arr_time = next_arr.time

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
      _, _, node_id = self.current_reserve_jobs.pop(0)
      self._reserve_nodes[node_id].remove_job(next_dep.job_id)
      return next_event
    else:
      next_event = Event(etype.schedule, next_arr_time, next_arr.job_id)
      self.next_jobID += 1
      return next_event

  class LJWStrategy(Enum):
    BASELINE = 1
    MLRUNTIMEMODEL = 2
    SPECULATIVEEXECUTION = 3

  """
  Run the simulator (event based) with waiting queue. Applicable to LJW and compound policy.
  """

  def run_LJW(
          self,
          ljw_strategy: LJWStrategy,
          max_wait_time_min: int,
          short_thresh_min: int,
          duration_hrs: int = None,
  ) -> tuple:
    # start time of simulation; for measuring the total simulation time
    t_start = time.time()

    # bookkeeping
    reserve_cpu_time = 0
    on_demand_count = 0

    # Next event for the simulator
    next_event = self._get_next_event()

    # For computing cost
    start = end = self.input_trace.iloc[0].time

    while next_event:
      if next_event.etype == etype.schedule:
        # New job request
        cur_job = self.input_trace.iloc[next_event.job_id]
        cur_time = next_event.time

        # Job requirements
        cpu_req = cur_job.num_cores
        runtime_sec = cur_job.wallclock_runtime_sec if ljw_strategy == Simulator.LJWStrategy.BASELINE else cur_job.predicted_time

        if runtime_sec <= short_thresh_min * 60:
          # Run the job on on-demand -- Short job
          on_demand_count += 1
          self.job_schedule_map[cur_job.job_id] = "D"
        elif (not self.wait_queue) and self._schedule_job(
                cur_time, cpu_req, runtime_sec, next_event.job_id
        ):
          # Book keeping
          reserve_cpu_time += cpu_req * runtime_sec
          end = max(end, cur_time + timedelta(seconds=int(runtime_sec)))
        else:
          # Add job to wait queue
          self.wait_queue.append(cur_job.job_id)
      else:
        # A job is finished. So, check the wait queue
        while len(self.wait_queue) > 0:
          job_id = self.wait_queue[0]
          waiting_job = self.input_trace.iloc[job_id]
          time_waited_sec = (
                  next_event.time - waiting_job.time
          ).total_seconds()

          # Compute maximum balking time
          max_wait_time_sec = max_wait_time_min * 60

          # Check if job waited more than the balking time
          if time_waited_sec > max_wait_time_sec:
            # Remove the job from the queue
            self.wait_queue.popleft()
            self.wait_time_map[job_id] = max_wait_time_sec
            self.job_schedule_map[job_id] = "D"
            on_demand_count += 1

            end = max(
              end, next_event.time + timedelta(seconds=int(runtime_sec))
            )
          else:
            # Try scheduling the job
            cpu_req = waiting_job.num_cores
            runtime_sec = waiting_job.wallclock_runtime_sec
            if self._schedule_job(
                    next_event.time, cpu_req, runtime_sec, job_id
            ):
              # Book keeping
              reserve_cpu_time += cpu_req * runtime_sec

              # remove the job from the queue
              self.wait_queue.popleft()

              # Add waiting time to wait times map
              self.wait_time_map[job_id] = time_waited_sec

              end = max(
                end,
                next_event.time + timedelta(seconds=int(runtime_sec)),
                )
            else:
              break
      next_event = self._get_next_event()

    t_end = time.time()
    print("Time taken in seconds is ", t_end - t_start)

    # Post- Simulation: Add schedule column and wait_time column to input dataframe
    self.input_trace["schedule"] = (
      self.input_trace["job_id"].map(self.job_schedule_map).fillna("D")
    )
    self.input_trace["wait_time_sec"] = (
      self.input_trace["job_id"].map(self.wait_time_map).fillna(0)
    )

    # Steady state
    h = 0.1  # head
    steady_state = self.input_trace[
      (self.input_trace.job_id >= h * self.job_count)
      & (self.input_trace.job_id <= (1.0 - h) * self.job_count)
      ]
    mean_wait_time_sec = steady_state[
      steady_state.schedule == "R"
      ].wait_time_sec.mean()
    balking_rate = (
            1
            - self.input_trace[self.input_trace.schedule == "R"].time.count()
            / self.input_trace.time.count()
    )

    # LJW and Compound policy -- On-demand jobs don't wait in queue
    mean_wait_time_sec = mean_wait_time_sec * (1 - balking_rate)

    # Simulation time
    elapsed_simulation_time = (end - start).total_seconds()

    # User has not specified trace duration
    if duration_hrs is None:
      duration_hrs = elapsed_simulation_time / 3600

    # Computing total cost
    (
      total_cost,
      reserved_cost,
      on_demand_cost,
      effective_price,
    ) = self._compute_total_cost(self.input_trace, duration_hrs)
    print(
      "Reserved cost is ",
      reserved_cost,
      ", On demand cost is ",
      on_demand_cost,
      ", Total cost is ",
      reserved_cost + on_demand_cost,
      )
    print("----------------------------------------------------------------")

    return (mean_wait_time_sec, effective_price, balking_rate, total_cost)

if __name__ == "__main__":
  # Trinity
  input_trace = pd.read_csv(os.path.abspath('../Dataset/TrinityPredictionsRandomForestTest.csv'))
  input_trace['time'] = pd.to_datetime(input_trace['time'])

  print("running Trinity LJW baseline")
  new_sim = Simulator(NUM_VMS=9408, VM_CPU=32, input_trace=input_trace)
  print(new_sim.run_LJW(ljw_strategy=Simulator.LJWStrategy.BASELINE, max_wait_time_min=1440, short_thresh_min=3))

  print("running Trinity LJW Random Forest Prediction")
  new_sim = Simulator(NUM_VMS=9408, VM_CPU=32, input_trace=input_trace)
  print(new_sim.run_LJW(ljw_strategy=Simulator.LJWStrategy.MLRUNTIMEMODEL, max_wait_time_min=1440, short_thresh_min=3))

  print("running Trinity LJW Speculative Execution")
  new_sim = Simulator(NUM_VMS=9408, VM_CPU=32, input_trace=input_trace)
  print(new_sim.run_LJW(ljw_strategy=Simulator.LJWStrategy.SPECULATIVEEXECUTION, max_wait_time_min=1440, short_thresh_min=3))

  # Mustang
  input_trace = pd.read_csv(os.path.abspath('../Dataset/MustangPredictionsRandomForestTest.csv'))
  input_trace['time'] = pd.to_datetime(input_trace['time'])

  print("running Mustang LJW baseline")
  new_sim = Simulator(NUM_VMS=1600, VM_CPU=24, input_trace=input_trace)
  print(new_sim.run_LJW(ljw_strategy=Simulator.LJWStrategy.BASELINE, max_wait_time_min=1440, short_thresh_min=3))

  print("running Mustang LJW Random Forest Prediction")
  new_sim = Simulator(NUM_VMS=1600, VM_CPU=24, input_trace=input_trace)
  print(new_sim.run_LJW(ljw_strategy=Simulator.LJWStrategy.MLRUNTIMEMODEL, max_wait_time_min=1440, short_thresh_min=3))

  print("running Mustang LJW Speculative Execution")
  new_sim = Simulator(NUM_VMS=1600, VM_CPU=24, input_trace=input_trace)
  print(new_sim.run_LJW(ljw_strategy=Simulator.LJWStrategy.SPECULATIVEEXECUTION, max_wait_time_min=1440, short_thresh_min=3))