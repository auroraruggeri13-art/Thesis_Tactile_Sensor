# Pipeline Execution Guide

Quick reference for running the two sensing pipelines end-to-end.
For full details see [README.md](README.md).

---

## Part 1 — Barometer-Based Tactile Sensing

Before running any script, update `config.py` with your local data path.

```
Step 1  Collect raw data  (hardware)
          -> apriltag_detections_trial<N>.txt
          -> ati_middle_trial<N>.txt
          -> barometers_trial<N>.txt

Step 2  Extract & synchronize poses + forces
        Script: Barometers_Based_Tactile_Sensor/Pose_Forces_Synchronized_with_Baro_Processing.py
        Input:  test data/test <N> - sensor v<V>/{apriltag, ati, barometers} files
        Output: test data/test <N> - sensor v<V>/synchronized_events_<N>.csv
        Config: TEST_NUMS list at top of script

Step 3  Build train / validation / test datasets
        Script: Barometers_Based_Tactile_Sensor/train_validation_test_dataset_generation.py
        Input:  list of synchronized_events_<N>.csv files
        Output: train_validation_test_data/train_data_v<V>.csv
                train_validation_test_data/validation_data_v<V>.csv
                train_validation_test_data/test_data_v<V>.csv
        Config: CSV_FILENAMES list, SENSOR_VERSION

Step 4  Train ML model  (choose one)
        [RECOMMENDED]
        Script: Barometers_Based_Tactile_Sensor/LightGBM_sliding_window_predictions.py
        Input:  train/val/test CSVs from Step 3
        Output: models parameters/averaged models/lightgbm_sliding_window_model_v<V>.pkl
                models parameters/averaged models/scaler_sliding_window_v<V>.pkl

        [Alternatives]
        linear:      Linear_Regression_force_predictions.py
        rf:          random_forest_force_prediction.py
        mlp:         NN_force_prediction.py
        lstm:        LSTM.py
        cnn:         1D_CNN_prediction.py
        transformer: transformer_sliding_window_predictions.py

Step 5  Real-time inference  (no ROS needed)
        Script: realtime_demo.py
        Input:  Arduino via serial port + trained model from Step 4
        Output: live predictions printed to terminal
        Run:    python realtime_demo.py --port COM3
```

---

## Part 2 — Vision-Based Force Estimation

```
Step 1  Extract frames from ROS bag
        Script: Vision_vs_Tactile/direct_jpeg_extractor.py
        Input:  ROS bag file with camera + barometer topics
        Output: frames_<bag_name>/*.jpg
                extracted_data.csv

Step 2  Estimate forces from deformation
        Script: Vision_vs_Tactile/Codes_for_cluster/code/forces_for_free.py
        Input:  extracted frames + synchronized CSV from Step 1
        Output: predicted forces CSV + comparison plots

        [HPC cluster]
        Scripts: Vision_vs_Tactile/Codes_for_cluster/run_forces.sh
                 Vision_vs_Tactile/Codes_for_cluster/run_forces_fusion.sh
```

---

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `Barometers_Based_Tactile_Sensor/force_resolution_static_analysis.py` | Compute force resolution from quasi-static intervals |
| `Barometers_Based_Tactile_Sensor/hysteresis_material_testing.py` | Characterize material hysteresis |
| `Barometers_Based_Tactile_Sensor/Sensor1_Repeteability.py` | Cross-sensor repeatability analysis |
| `Barometers_Based_Tactile_Sensor/COMSOL_pressure_analysis.py` | Analyze COMSOL FEA simulation outputs |
| `Barometers_Based_Tactile_Sensor/raw_barometers_plot.py` | Quick barometer signal visualization |
| `Vision_vs_Tactile/plot_models_predictions_from_csv.py` | Compare vision vs. tactile vs. fusion predictions |
