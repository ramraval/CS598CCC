"""
Copyright (c) 2020 <Sustainable Computing Lab>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import sys
import pandas as pd
import warnings
from Simulator import Simulator

warnings.simplefilter(action='ignore', category=FutureWarning)

RUNTIME_THRESHOLD_VALUES = [5, 15, 60]
WAITING_TIME_THRESHOLD_VALUES = [3, 6, 12]

"""
Runs simulator for specified technique and outputs results
"""

def get_results(simulator, dataset, runtime_threshold, waiting_time_threshold, technique) -> None:
    match technique:
        case 'custom':
            print("Running custom approach on " + dataset + " with runtime threshold of " + str(runtime_threshold) +
                  " minutes and waiting time threshold of " + str(waiting_time_threshold) + " hour(s)")
            simulator.run_custom_approach(runtime_threshold_min=runtime_threshold,
                                          waiting_time_threshold_hour=waiting_time_threshold)
        case 'original':
            print("Running original approach on " + dataset + " with runtime threshold of " + str(runtime_threshold) +
                  " minutes and waiting time threshold of " + str(waiting_time_threshold) + " hour(s)")
            simulator.run_original(runtime_threshold_min=runtime_threshold,
                                   waiting_time_threshold_hour=waiting_time_threshold)
        case 'AJW':
            print("Running AJW on " + dataset)
            simulator.run_AJW()
        case 'NJW':
            print("Running NJW on " + dataset)
            simulator.run_NJW()

if __name__ == "__main__":
    dataset = sys.argv[1]
    technique = sys.argv[2]

    if dataset == 'ANL':
        input_trace = pd.read_csv(os.path.abspath('../Dataset/ANL_trace_output_FINAL.csv')) \
            .sort_values(by=['submit_time'])
        training_trace = pd.read_csv(os.path.abspath('../Dataset/cleaned_ANL_with_waiting_times_full.csv'))
        num_VMs = 40960
        vm_CPU = 4
    elif dataset == 'RICC':
        input_trace = pd.read_csv(os.path.abspath('../Dataset/RICC_trace_output_FINAL.csv')) \
            .sort_values(by=['submit_time'])
        training_trace = pd.read_csv(os.path.abspath('../Dataset/cleaned_RICC_with_waiting_times_full.csv'))
        num_VMs = 1024
        vm_CPU = 8
    else:
        exit(1)

    simulator = Simulator(dataset=dataset, NUM_VMS=num_VMs, VM_CPU=vm_CPU, input_trace=input_trace,
                          training_trace=training_trace)
    for runtime_threshold in RUNTIME_THRESHOLD_VALUES:
        for waiting_time_threshold in WAITING_TIME_THRESHOLD_VALUES:
            simulator = Simulator(dataset=dataset, NUM_VMS=num_VMs, VM_CPU=vm_CPU, input_trace=input_trace,
                                  training_trace=training_trace)
            get_results(simulator, dataset, runtime_threshold, waiting_time_threshold, technique)
