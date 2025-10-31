#!/bin/bash

number_of_jobs=2
MEMORY_LIMIT="40G"
locations=20
axis_positions=36
progress_file="progress_axis_position/current_batch.txt"

# Limit memory usage to 40 GB (u kilobajtima)
ulimit -v $((40 * 1024 * 1024))

core_id=-1
count=0
batch_number=1

mkdir -p progress_axis_position

for ((i=0; i<locations; i++)); do
  for ((j=0; j<axis_positions; j++)); do
    core_id=$(( (core_id + 1) % number_of_jobs ))  # rotira kroz raspoloživa jezgra

    input_file="input_axis_position/input_${i}_${j}.py"
    drift_file="output_axis_position/drift_precession_${i}_${j}.txt"
    prog_file="progress_axis_position/progress_precession_${i}_${j}.log"
    nohup_file="progress_axis_position/nohup_${i}_${j}.out"

    echo "Running $input_file on core $core_id"

    nohup taskset -c ${core_id} \
      python3 ../diurnal_yarkovsky_effect.py \
      -input "$input_file" \
      -yarko "$drift_file" \
      -prog "$prog_file" \
      > "$nohup_file" 2>&1 &

    ((count++))

    # kada dostigneš limit od number_of_jobs, čekaj da se svi završe
    if (( count % number_of_jobs == 0 )); then
      echo ">>> Waiting for batch #$batch_number ($number_of_jobs jobs) to finish..."
      echo "$batch_number" > "$progress_file"
      wait
      echo ">>> Batch #$batch_number finished. Starting next set..."
      ((batch_number++))
    fi
  done
done

# sačekaj i poslednju grupu (ako nije cela)
wait
echo "$batch_number" > "$progress_file"
echo ">>> All jobs completed. Last batch: $batch_number."
