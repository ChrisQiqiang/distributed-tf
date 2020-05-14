#!/bin/bash

/usr/bin/python /image/tensorflow-resnet/cifar10_resnet_sync.py --task_index=$1 --job_name=$2 --worker_hosts=$3 --ps_hosts=$4 --sync=1 &
cd /
./getnetinfo 
sleep 10000
