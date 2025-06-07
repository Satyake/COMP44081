# COMP44081

Data Splitting

Training set: all but the final year of each fold.

Validation set: the last 730 days of the training set (equal to input_chunk_length + output_chunk_length) ensure the model sees enough points to compute both look-back and forecast windows.

Test set: the final 365 days, reserved strictly for out-of-sample evaluation.

Preprocessing

Each series is wrapped in Darts’ TimeSeries object.

A Scaler transformer fits on the training series and then transforms the training and validation series, normalizing magnitude differences and improving convergence.

Model Architecture

Utilizes N-BEATS, a purely deep-learning forecasting architecture that stacks backward and forward residual blocks to capture complex temporal patterns without manual feature engineering.

Hyperparameters:

input_chunk_length=365 (one year of history per prediction window)

output_chunk_length=365 (forecast horizon of one year)

n_epochs=50 (with optional early stopping if enabled)

random_state=43 for reproducibility

Training runs on GPU for efficiency.

Early Stopping

(Optional) A PyTorch-Lightning EarlyStopping callback monitors validation loss (val_loss) and halts training after patience epochs with no improvement, preventing overfitting and saving compute.

Forecasting & Post-processing

After training, the model predicts the next 365 days on the normalized scale.

Inverse transform returns forecasts to original units.

A final clamping step sets any negative forecasts to zero:
