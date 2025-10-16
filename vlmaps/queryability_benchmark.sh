#!/bin/bash
./predict_map_all.sh
./postprocess_predicted_all.sh
./measure_classification_all.sh
./measure_instance_classification.sh