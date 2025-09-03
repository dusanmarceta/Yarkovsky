#!/bin/bash

first_core=0
number_of_jobs=37
MEMORY_LIMIT="40G"

# Limit memory usage to 40 GB (u kilobajtima)
ulimit -v $((40 * 1024 * 1024))

for ((i=0; i<number_of_jobs; i++)); do
    core_id=$((first_core + i))   # svaka iteracija ide na svoje jezgro

    input_file="input/input_precession_${i}.py"
    drift_file="output/drift_precession_${i}.txt"
    prog_file="progress/progress_precession_${i}.log"
    nohup_file="progress/nohup_${i}.out"

    echo "Running $input_file on core $core_id"

    nohup taskset -c ${core_id} \
    python3 ../diurnal_yarkovsky_effect.py \
    -input "$input_file" \
    -yarko "$drift_file" \
    -prog "$prog_file" \
    > "$nohup_file" 2>&1 &
done

