#!/bin/bash

python weight_virtualization.py -mode=t -vnn_name=network1 -iter=20000
python weight_virtualization.py -mode=e -vnn_name=network1
python weight_virtualization.py -mode=e -vnn_name=network2
python weight_virtualization.py -mode=e -vnn_name=network3
python weight_virtualization.py -mode=e -vnn_name=network4
python weight_virtualization.py -mode=e -vnn_name=network5

python weight_virtualization.py -mode=t -vnn_name=network2 -iter=20000
python weight_virtualization.py -mode=e -vnn_name=network1
python weight_virtualization.py -mode=e -vnn_name=network2
python weight_virtualization.py -mode=e -vnn_name=network3
python weight_virtualization.py -mode=e -vnn_name=network4
python weight_virtualization.py -mode=e -vnn_name=network5

python weight_virtualization.py -mode=t -vnn_name=network3 -iter=20000
python weight_virtualization.py -mode=e -vnn_name=network1
python weight_virtualization.py -mode=e -vnn_name=network2
python weight_virtualization.py -mode=e -vnn_name=network3
python weight_virtualization.py -mode=e -vnn_name=network4
python weight_virtualization.py -mode=e -vnn_name=network5

python weight_virtualization.py -mode=t -vnn_name=network4 -iter=20000
python weight_virtualization.py -mode=e -vnn_name=network1
python weight_virtualization.py -mode=e -vnn_name=network2
python weight_virtualization.py -mode=e -vnn_name=network3
python weight_virtualization.py -mode=e -vnn_name=network4
python weight_virtualization.py -mode=e -vnn_name=network5

python weight_virtualization.py -mode=t -vnn_name=network5 -iter=20000
python weight_virtualization.py -mode=e -vnn_name=network1
python weight_virtualization.py -mode=e -vnn_name=network2
python weight_virtualization.py -mode=e -vnn_name=network3
python weight_virtualization.py -mode=e -vnn_name=network4
python weight_virtualization.py -mode=e -vnn_name=network5
