#!/bin/bash

/usr/bin/python /image/Distribute_MNIST/distributed.py --task_index=$1 --job_name=$2 --worker_hosts=$3 --ps_hosts=$4 --issync=0 &
cd /
./getnetinfo 
sleep 10000
