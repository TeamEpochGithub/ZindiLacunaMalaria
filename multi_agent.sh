#!/bin/bash

for i in {1..16}; do
    (
        wandb agent ignacekonig-epoch-tu-delft/final_pp_sweep/yt7x9f3t > /dev/null 2>&1
    ) &
done

wait
