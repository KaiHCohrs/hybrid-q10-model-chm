# hybrid-q10-model
Hybrid modeling approaches for Q10 estimation from the "Double machine learning for causal hybrid model - applications in the Earth sciences" manuscript.

1. Download the 2015 halfhourly Fullset data from site AT-Neu and store the csv in data.
2. Run DMLHM_Q10.py or GDHM_Q10.py respectively for 100 repetitions of the DML-based hybrid modeling or the GD-based hybrid modeling respectively. 

Note that the GD-based model is fully implemented in jax and hence runs in parallel while the 100 runs of the DML-based approach runs the 100 iterations sequentially. 