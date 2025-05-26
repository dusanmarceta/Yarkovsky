for i_ast in $(seq 0 1); do
    for i_rho in $(seq 0 2); do
        for i_k in $(seq 0 2); do
            for i_D in $(seq 0 2); do
		    # Use the range of cores with taskset
		    nohup python3 seasonal_PT_GE1.py "$i_ast" "$i_rho" "$i_k" "$i_D" &
            done
        done
    done
done
