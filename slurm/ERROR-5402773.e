2025-07-29 12:39:13.905337: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-07-29 12:39:14.553854: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1753807154.780757 3066982 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1753807154.838723 3066982 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1753807155.337850 3066982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753807155.337904 3066982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753807155.337907 3066982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1753807155.337910 3066982 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2025-07-29 12:39:15.389591: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-07-29 12:39:24,712 INFO Loaded data: 93612 rows, 18 columns.
2025-07-29 12:39:24,713 INFO Environment state_space: 291, stock_dim: 29
2025-07-29 12:39:24,724 INFO Training A2C agent...
Traceback (most recent call last):
  File "/deac/csc/classes/csc790-sp-2025/jianb21/Ensemble_stockTrading_2020/scripts/main.py", line 250, in <module>
    main()
  File "/deac/csc/classes/csc790-sp-2025/jianb21/Ensemble_stockTrading_2020/scripts/main.py", line 240, in main
    train_a2c_agent(agent)
  File "/deac/csc/classes/csc790-sp-2025/jianb21/Ensemble_stockTrading_2020/scripts/main.py", line 143, in train_a2c_agent
    trained_a2c = agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=10_000_000) # 10 million
  File "/deac/csc/classes/csc790-sp-2025/jianb21/Ensemble_stockTrading_2020/agents/DRLAgent.py", line 255, in train_model
    model = model.learn(
  File "/home/jianb21/.local/lib/python3.9/site-packages/stable_baselines3/a2c/a2c.py", line 201, in learn
    return super().learn(
  File "/home/jianb21/.local/lib/python3.9/site-packages/stable_baselines3/common/on_policy_algorithm.py", line 335, in learn
    self.dump_logs(iteration)
  File "/home/jianb21/.local/lib/python3.9/site-packages/stable_baselines3/common/on_policy_algorithm.py", line 298, in dump_logs
    self.logger.dump(step=self.num_timesteps)
  File "/home/jianb21/.local/lib/python3.9/site-packages/stable_baselines3/common/logger.py", line 540, in dump
    _format.write(self.name_to_value, self.name_to_excluded, step)
  File "/home/jianb21/.local/lib/python3.9/site-packages/stable_baselines3/common/logger.py", line 382, in write
    self.file.flush()
OSError: [Errno 116] Stale file handle
