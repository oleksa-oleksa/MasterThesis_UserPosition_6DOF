#!/bin/bash

while read line
do
   echo "Record is : $line"
done < <(tail -n +2 $1)

#export VAR=abc