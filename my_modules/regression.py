import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def regression_app():
    st.header("📈 Regression Module")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Dataset Preview with Statistics
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Preview:")
            st.dataframe(df.head())
        with col2:
            st.write("Dataset Statistics:")
            st.dataframe(df.describe())
            
        # Data Preprocessing Options
        st.subheader("🔧 Data Preprocessing")
        preprocessing_options = st.expander("Preprocessing Options")
        with preprocessing_options:
            # Handle missing values
            handle_missing = st.checkbox("Handle Missing Values")
            if handle_missing:
                missing_strategy = st.selectbox(
                    "Missing Values Strategy",
                    ["Drop", "Mean", "Median", "Zero"]
                )
                if missing_strategy == "Drop":
                    df = df.dropna()
                elif missing_strategy == "Mean":
                    df = df.fillna(df.mean())
                elif missing_strategy == "Median":
                    df = df.fillna(df.median())
                elif missing_strategy == "Zero":
                    df = df.fillna(0)
            
            # Feature scaling
            scaling_option = st.checkbox("Scale Features")
            if scaling_option:
                scaler_type = st.selectbox(
                    "Scaling Method",
                    ["StandardScaler", "MinMaxScaler"]
                )
                if scaler_type == "StandardScaler":
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
        
        # Feature and target selection
        target = st.selectbox("Select Target Column (e.g., Price)", df.columns)
        features = st.multiselect("Select Feature Columns (e.g., Size, Bedrooms)", 
                                [col for col in df.columns if col != target])
        
        if features and target:
            X = df[features]
            y = df[target]
            
            # Apply scaling if selected
            if scaling_option:
                X = pd.DataFrame(scaler.fit_transform(X), columns=features)
            
            # Train-test split
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Model training
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Model evaluation
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance Metrics")
                metrics = {
                    "Training R² Score": r2_score(y_train, train_predictions),
                    "Test R² Score": r2_score(y_test, test_predictions),
                    "Test MSE": mean_squared_error(y_test, test_predictions),
                    "Test MAE": mean_absolute_error(y_test, test_predictions),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, test_predictions))
                }
                
                metrics_df = pd.DataFrame({
                    "Metric": list(metrics.keys()),
                    "Value": [round(v, 3) for v in metrics.values()]
                })
                st.table(metrics_df)
            
            with col2:
                st.subheader("Feature Importance")
                coef_df = pd.DataFrame({
                    'Feature': features,
                    'Coefficient': model.coef_
                })
                fig_importance = px.bar(coef_df, x='Feature', y='Coefficient',
                                      title='Feature Coefficients')
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Enhanced Visualization Section
            st.subheader("📈 Model Visualizations")
            
            # Feature Relationships and Regression Line
            if len(features) == 1:  # Single feature case
                fig_reg = px.scatter(
                    df, x=features[0], y=target,
                    trendline="ols",
                    title=f"Regression Line: {features[0]} vs {target}",
                    template="plotly_white"
                )
                fig_reg.update_layout(
                    xaxis_title=features[0],
                    yaxis_title=target,
                    showlegend=True,
                    height=500
                )
                fig_reg.update_traces(
                    marker=dict(size=8, opacity=0.6),
                    name="Data Points"
                )
                fig_reg.data[1].update(name="Regression Line")
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                # Feature correlation heatmap for multiple features
                correlation_matrix = df[features + [target]].corr()
                fig_corr = px.imshow(
                    correlation_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    labels=dict(color="Correlation")
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Scatter plot of predictions vs actual values
            st.subheader("Predictions vs Actual Values")
            fig_scatter = go.Figure()
            
            # Training data
            fig_scatter.add_trace(go.Scatter(
                x=y_train,
                y=train_predictions,
                mode='markers',
                name='Training Data',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            
            # Test data
            fig_scatter.add_trace(go.Scatter(
                x=y_test,
                y=test_predictions,
                mode='markers',
                name='Test Data',
                marker=dict(color='red', size=8, opacity=0.6)
            ))
            
            # Perfect prediction line
            min_val = min(min(y_train), min(y_test))
            max_val = max(max(y_train), max(y_test))
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='green', dash='dash')
            ))
            
            fig_scatter.update_layout(
                title='Predicted vs Actual Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                showlegend=True
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Enhanced Prediction Interface
            st.subheader("🎯 Make Predictions")
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                st.write("Enter Feature Values:")
                input_data = {}
                for feature in features:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_data[feature] = st.slider(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Range: {min_val:.2f} to {max_val:.2f}"
                    )
            
            with pred_col2:
                st.write("Bulk Prediction:")
                sample_csv = pd.DataFrame(columns=features).to_csv(index=False)
                st.download_button(
                    "Download Template CSV",
                    sample_csv,
                    "prediction_template.csv",
                    "text/csv"
                )
                bulk_file = st.file_uploader("Upload Prediction CSV", type=["csv"])
                
            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                if scaling_option:
                    input_df = pd.DataFrame(scaler.transform(input_df), columns=features)
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted {target}: {prediction:.2f}")
                
            if bulk_file:
                bulk_df = pd.read_csv(bulk_file)
                if scaling_option:
                    bulk_df = pd.DataFrame(scaler.transform(bulk_df), columns=features)
                bulk_predictions = model.predict(bulk_df)
                result_df = pd.DataFrame({
                    "Prediction": bulk_predictions
                })
                st.write("Bulk Predictions:")
                st.dataframe(result_df)
                
                # Download predictions
                csv = result_df.to_csv(index=False)
                st.download_button(
                    "Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
