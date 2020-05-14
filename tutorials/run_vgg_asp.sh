#!/bin/bash

/usr/bin/python /image/tensorflow-vgg/cifar10_vgg_sync.py --task_index=$1 --job_name=$2 --worker_hosts=$3 --ps_hosts=$4 --sync=0 &
cd /
./getnetinfo
sleep 10000
