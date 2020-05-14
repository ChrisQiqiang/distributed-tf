master='192.168.2.200:22234'
worker_hosts=''
slaves='192.168.2.202'
#	192.168.2.202'
for i in $slaves
do
        worker_hosts=$worker_hosts$i":22234,"
done
worker_hosts=${worker_hosts%%,}
echo $worker_hosts
nohup python ~/resnet/cifar10_resnet_sync.py  --job_name=ps --task_id=0 --ps_hosts=$master --worker_hosts=$worker_hosts  >master.log 2>&1 &
num=0
for i in $slaves
do
nohup ssh $i "python ~/resnet/cifar10_resnet_sync.py  --job_name=worker --task_id=$num --ps_hosts=$master --worker_hosts=$worker_hosts" >slave$num.log 2>&1 &
num=$((num+1))
done
