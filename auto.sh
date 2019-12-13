#!/bin/bash

for i in {1..100}
do
   	echo "Welcome $i times"
	python virtual_brain.py -mode=t -vnn_name=network1 -iter=2000
	python virtual_brain.py -mode=t -vnn_name=network2 -iter=2000
	python virtual_brain.py -mode=t -vnn_name=network3 -iter=2000
	python virtual_brain.py -mode=t -vnn_name=network4 -iter=2000
	python virtual_brain.py -mode=t -vnn_name=network5 -iter=2000
done
