# Implementation of event based simulator

This simulator modifies the simulator provided by the Sustainable Computing Lab at the following link: https://github.com/sustainablecomputinglab/waitinggame

Per the Sustainable Computing Lab, the simulator can be described as follows: "We implemented a trace-driven job simulator in python that mimics a cloud-enabled job scheduler, which can acquire VMs on-demand to service jobs. The simulator uses a FCFS scheduling policy, and also implements each of our waiting policies."

We use this simulator to compare the implied cost and mean waiting time of our custom waiting policy implementation versus a traditional implementation. 

## To run the simulator (run_simulator.py)

Specify the dataset ('ANL' or 'RICC') and the approach ('custom', 'original', 'NJW', 'AJW')

Note the input training trace for RICC has not been uploaded to the GitHub repository due to size limitations
