for i in $slaves
do
	ssh $i "pkill python"
	echo "finish clear!"$i
done

