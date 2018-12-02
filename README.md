# Motion Classification Using Turicreate
Follow up instructions https://apple.github.io/turicreate/docs/userguide/activity_classifier/

## Data Structure:
Matrix(N*M) where N = E * T where E: number of samples of both drift and no drift and T: the time window. M: {acc_x, acc_y, acc_z, rotation_x, rotation_y, rotation_z, exp_id, activity}

## HAPT Data
HAPT is the sample provided by apple at the following link https://apple.github.io/turicreate/docs/userguide/activity_classifier/

## Drift
Use the Jupyter notebook "Turicreate Drift" which has further information of the data

### Drift_Data
contains the sample data

### DriftClassifier.mlmodel
The testing CoreML model.

## ViewController
The swift which utilizes both Core Motion and CoreML
