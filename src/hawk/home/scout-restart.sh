#!/bin/bash
###############################################################################
#Script Name    : scout-restart.sh
#Description    : execute restart commands on multiple scouts
################################################################################

fireqos stop
sudo service hawk stop
conda activate hawk
cd $HOME/hawk
python setup.py install
sudo service hawk start

exit 0
