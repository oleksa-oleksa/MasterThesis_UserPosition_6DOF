#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
FILES="Obj\JOSH\*"
for f in $FILES
do
	  echo "Processing $f file..."
	  obj2gltf -i $f -o "${f%.*}".gltf
      done
