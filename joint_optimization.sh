#!/bin/bash

for i in {1..10}
do
   	echo "Welcome $i times"
	python weight_virtualization.py -mode=t -vnn_name=network1 -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=network1
	python weight_virtualization.py -mode=e -vnn_name=network2
	python weight_virtualization.py -mode=e -vnn_name=network3
	python weight_virtualization.py -mode=e -vnn_name=network4
	python weight_virtualization.py -mode=e -vnn_name=network5

	python weight_virtualization.py -mode=t -vnn_name=network2 -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=network1
	python weight_virtualization.py -mode=e -vnn_name=network2
	python weight_virtualization.py -mode=e -vnn_name=network3
	python weight_virtualization.py -mode=e -vnn_name=network4
	python weight_virtualization.py -mode=e -vnn_name=network5

	python weight_virtualization.py -mode=t -vnn_name=network3 -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=network1
	python weight_virtualization.py -mode=e -vnn_name=network2
	python weight_virtualization.py -mode=e -vnn_name=network3
	python weight_virtualization.py -mode=e -vnn_name=network4
	python weight_virtualization.py -mode=e -vnn_name=network5

	python weight_virtualization.py -mode=t -vnn_name=network4 -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=network1
	python weight_virtualization.py -mode=e -vnn_name=network2
	python weight_virtualization.py -mode=e -vnn_name=network3
	python weight_virtualization.py -mode=e -vnn_name=network4
	python weight_virtualization.py -mode=e -vnn_name=network5

	python weight_virtualization.py -mode=t -vnn_name=network5 -iter=2000
	python weight_virtualization.py -mode=e -vnn_name=network1
	python weight_virtualization.py -mode=e -vnn_name=network2
	python weight_virtualization.py -mode=e -vnn_name=network3
	python weight_virtualization.py -mode=e -vnn_name=network4
	python weight_virtualization.py -mode=e -vnn_name=network5


done
