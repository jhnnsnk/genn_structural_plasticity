#!/bin/bash
readarray -t LINES < arguments.txt

NUM_LINES=${#LINES[@]}
echo "NUM_LINES: $NUM_LINES"
for i in $(seq 0 2 $NUM_LINES) 
do
    python classifier.py --train --device-id 0 ${LINES[$i]}  &
    python classifier.py --train --device-id 1 ${LINES[$i + 1]} &
    wait
    #python classifier.py --device-id 0 ${LINES[$i]}
    #python classifier.py --device-id 1 ${LINES[$i + 1]}
    #wait
done
