# Volume Forecasting (Modified with Holt-Winter Methods)
The original code provided does not include trend and seasonality, and uses the standard stats model library.
If we inspect the data, we see that there is a  strong seasonality component and trend when we decompose it.

The original model did not have a trend component in it.

I have used Exponential Smoothing with the Holt-Winters method, which accounts for both trend and seasonality as an additive component.
This resulted in a  good forecast of 365 days on the test horizon.

**HoltWinters with trend and seasonality.ipynb** -> This has the full code 

**Performance**:  {'Base': {0: 1.529857106512923,
  1: 0.8931257429195898,
  2: 0.8565136047622175,
  3: 0.849004666557113,
  4: 0.8623397604001105},
 'Base_modified': {0: 0.6864071360363305,
  1: 1.1035446150422632,
  2: 0.6599292290207176,
  3: 1.1365194991888812,
  4: 0.8280584686316488}}

  
**Model_updated.py** has the new model.

# Daily Sales Forecasting with N-BEATS (Proposed Model)
Reference library used: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html#darts.models.forecasting.nbeats.NBEATSModel

## Overview
- For forecasting the  **daily sales** I have used the very popular  N-BEATS Model (deep learning-based) [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting]
- Reason for choosing NBEATS: The model learns and  captures trends, seasonality, and short-term changes.
- The Model is in boosting model. Each submodel generates a forecast and a backcast with residuals, which are fed to the subsequent models.
- The Sub Model generates a 
    A backcast (an approximation of the input)
    A forecast (the predicted future)
– The backcast is subtracted from the block’s input to form a residual, which becomes the input to the next block.
  
## Flow of the code:

1. **Data Splitting**  
   - **Training**: all but the final 365 days of each fold.  
   - **Validation**: last `input_chunk_length + output_chunk_length` days (365 + 365 = 730). (This is not used here, but can improve the training performance by monitoring the validation loss.)  
   - **Test**: final 365 days, or Out of Time OOT 

2. **Preprocessing**  
   
   - The data has been resampled to a Daily frequency from the 15-minute differences.
   - I have scaled both the training and validation data.

3. **Model Architecture**  
   - **N-BEATSModel** with:
     - `input_chunk_length = 365` #This is the backcast #The model looks 365 days before and forecasts 365 days forward #NBEATS uses a backcast and forecast
     - `output_chunk_length = 365` 
     - `n_epochs = 50`  

    - I did use Early Stopping (Commented out Early Stopping, due to a bug in Paperspace)
    - The Final Model is the best-trained model with the optimized weights (Retrained with ES on Google colab).

4. **Forecasting & Post-processing**  
   ```python
   # raw forecast on normalized scale
   fc_scaled = model.predict(365) #predict next 1 year of daily sales (but on a daily number)
   # invert scale
   fc = scaler.inverse_transform(fc_scaled) (reversed scaled to get original data)
   # clamp negatives to zero
   fc_nonneg = fc.with_values(np.clip(fc.values(), 0, None)) (forecast values are unbounded, so this has been made range-bound to avoid negatives

5. ** Performance** (NBEATS)
[Fold1: 0.6961632335034433,
 Fold2: 2.508953940001475,
 Fold3: 0.6507345731317645,
 Fold4: 0.648070948148359,
 Fold5: 0.9550531186255872]

#Tools used: Used Paperspace GPU to train the NBEATS model on A4000 GPU, DARTS Documentation, and AWS Q Coding Assistant.
