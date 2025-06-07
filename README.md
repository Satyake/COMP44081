# Daily Redemption & Sales Forecasting Model

## Accessible Description
- Predicts **daily redemption volumes** (e.g. fund redemptions, voucher redemptions) and **daily sales** using historical data.  
- Built on a state-of-the-art deep learning architecture (N-BEATS) that learns seasonality, trend, and short-term fluctuations automatically.  
- Uses a **5-fold time-series split** with a one-year hold-out to ensure robust performance on unseen data.  
- Normalizes inputs via a scaler, then restores forecasts to the original scale for interpretability.  
- Clips negative outputs to zero so all predictions remain meaningful (you can’t sell or redeem a negative amount).  
- Measures accuracy with **RMSE** (root-mean-square error) to quantify average daily forecast error.  
- Produces easy-to-read plots overlaying train, actual, and forecast series, helping teams plan inventory, staffing, or cash-flow needs.

## Detailed Technical Description
The pipeline performs 5-fold cross-validation with a fixed 365-day test hold-out. For each fold:

1. **Data Splitting**  
   - **Training set**: all but the final 365 days of the fold.  
   - **Validation set**: last 730 days of training (equal to `input_chunk_length + output_chunk_length`) to support one look-back + one forecast window.  
   - **Test set**: final 365 days, held out for final evaluation.

2. **Preprocessing**  
   - Wraps each series in Darts’ `TimeSeries` object.  
   - Applies a `Scaler` fit on the training slice, then transforms both training and validation series for stable training.

3. **Model Architecture**  
   - **N-BEATS**: stacks backward/forward residual blocks to model complex temporal patterns without manual feature engineering.  
   - **Hyperparameters**:
     - `input_chunk_length = 365` (one year of history)  
     - `output_chunk_length = 365` (forecast horizon of one year)  
     - `n_epochs = 50`  
     - `random_state = 43`  
   - Training runs on GPU for performance.

4. **Early Stopping (optional)**  
   - A PyTorch-Lightning `EarlyStopping` callback monitors `val_loss` and stops training after `patience` epochs without improvement, preventing overfitting.

5. **Forecasting & Post-processing**  
   - Predicts the next 365 days on the normalized scale.  
   - Applies inverse scaling to return to original units.  
   - Clips any negative values to zero:
     ```python
     forecast_nonneg = forecast.with_values(np.clip(forecast.values(), 0, None))
     ```

6. **Evaluation & Visualization**  
   - Computes **RMSE** against the true test fold to measure average daily error.  
   - Generates plots overlaying train, actual, and forecast series for each fold, enabling quick diagnostic checks of trend capture and peak accuracy.

This framework can be applied to any daily time series—simply replace the input with your sales or redemption data to generate accurate, non-negative daily forecasts.
