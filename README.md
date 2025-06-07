# Volume Forecasting
The original code provided does not include trend and seasonality, and uses the standard stats model library.
If we inspect the data, we see that there is a  strong seasonality component and trend when we decompose it.

I have used Exponential Smoothing with the Holt-Winters method, which accounts for both trend and seasonality as an additive component.
This resulted in a  good forecast of 365 days on the test horizon.



# Daily Sales Forecasting with N-BEATS (Proposed Model)
Reference library used: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html#darts.models.forecasting.nbeats.NBEATSModel

## Overview
- Forecast **daily sales** using the N-BEATS Model (deep learning-based), which captures trends, seasonality, and short-term changes.
  
## Flow of the code:

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

4. **Forecasting & Post-processing**  
   ```python
   # raw forecast on normalized scale
   fc_scaled = model.predict(365) #predict next 1 year of daily sales
   # invert scale
   fc = scaler.inverse_transform(fc_scaled)
   # clamp negatives to zero
   fc_nonneg = fc.with_values(np.clip(fc.values(), 0, None))




















#Tools used: Github CoPilot, AWS Q
