# Daily Sales Forecasting with N-BEATS

## Overview
- Forecasts **daily sales** using a purely deep-learning architecture (N-BEATS) that automatically captures seasonality, trends, and short-term fluctuations.  
- Employs a **5-fold time-series split** with a fixed 365-day hold-out per fold to ensure robust generalization.  
- Normalizes inputs via a scaler and inversely transforms predictions back to original units for interpretability.  
- Clips any negative outputs to zero so all forecasts remain non-negative.  
- Evaluates accuracy using **RMSE** (root-mean-square error) on unseen test folds.  
- Provides visual plots overlaying training data, actuals, and forecasts for easy diagnostic review.

## Technical Details

1. **Data Splitting**  
   - **Training**: all but the final 365 days of each fold.  
   - **Validation**: last `input_chunk_length + output_chunk_length` days (365 + 365 = 730) carved from the end of the training fold for early stopping.  
   - **Test**: final 365 days, reserved for out-of-sample evaluation.

2. **Preprocessing**  
   - Wrap each series in Darts’ `TimeSeries`.  
   - Fit a `Scaler` on the training slice; transform both training and validation series.

3. **Model Architecture**  
   - **N-BEATSModel** with:
     - `input_chunk_length = 365`  
     - `output_chunk_length = 365`  
     - `n_epochs = 50`  
     - `random_state = 43`  
   - Trained on GPU; optional PyTorch-Lightning `EarlyStopping` (`monitor="val_loss"`, `patience=10`).

4. **Forecasting & Post-processing**  
   ```python
   # raw forecast on normalized scale
   fc_scaled = model.predict(365)
   # invert scale
   fc = scaler.inverse_transform(fc_scaled)
   # clamp negatives to zero
   fc_nonneg = fc.with_values(np.clip(fc.values(), 0, None))



#Tools used: Github CoPilot
