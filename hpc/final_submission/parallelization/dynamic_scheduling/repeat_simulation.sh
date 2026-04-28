#!/bin/bash

for n in {2..20}; do
  for i in {1..8}; do
    bsub -n "$n" < submit.sh
  done
done
