#!/bin/bash

for i in {1..8}; do
    bsub -n "$n" < submit.sh
done
