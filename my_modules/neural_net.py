# In my_modules/neural_net.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.express as px

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(FeedForward, self).__init__()
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, progress_bar, progress_text, scheduler=None):
    model.train()
    metrics = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        
        # Store metrics
        metrics['train_losses'].append(avg_train_loss)
        metrics['train_accuracies'].append(train_accuracy)
        
        # Update progress
        progress_text.text(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2%}')
        progress_bar.progress((epoch + 1) / num_epochs)
    
    return metrics

def make_prediction(model, features, scaler, label_encoder, device):
    model.eval()
    with torch.no_grad():
        # Scale features using the same scaler
        features_scaled = scaler.transform(features)
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        # Get predictions
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs.data, 1)
        # Convert back to original labels
        predictions = label_encoder.inverse_transform(predicted.cpu().numpy())
    return predictions

def neural_net_app():
    st.header("🤖 Neural Network Classification")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    hidden_sizes = [
        st.sidebar.slider("Hidden Layer 1 Size", 8, 128, 64),
        st.sidebar.slider("Hidden Layer 2 Size", 8, 128, 32)
    ]
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.1, 0.01, 0.001, 0.0001],
        value=0.01
    )
    batch_size = st.sidebar.select_slider(
        "Batch Size",
        options=[16, 32, 64, 128],
        value=32
    )
    num_epochs = st.sidebar.slider("Number of Epochs", 10, 100, 30)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Dataset Preview
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Preview:")
            st.dataframe(df.head())
        with col2:
            st.write("Dataset Statistics:")
            st.dataframe(df.describe())
        
        # Feature and target selection
        st.subheader("🎯 Feature and Target Selection")
        target_column = st.selectbox("Select Target Column", df.columns)
        feature_columns = st.multiselect(
            "Select Feature Columns",
            [col for col in df.columns if col != target_column],
            help="Select the columns to use as features"
        )
        
        if feature_columns and target_column:
            # Data preprocessing
            X = df[feature_columns]
            y = df[target_column]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Encode target
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            num_classes = len(label_encoder.classes_)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True
            )
            
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = FeedForward(
                input_size=len(feature_columns),
                hidden_sizes=hidden_sizes,
                num_classes=num_classes
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training progress
            st.subheader("📈 Training Progress")
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Train model with validation
            metrics = train_model(
                model, 
                train_loader,
                val_loader,
                criterion, 
                optimizer, 
                num_epochs, 
                device,
                progress_bar, 
                progress_text
            )
            
            # After training, add prediction interface
            st.subheader("🔮 Make Predictions")
            pred_tab1, pred_tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

            with pred_tab1:
                # Single prediction interface
                st.write("Enter values for prediction:")
                input_data = {}
                for feature in feature_columns:
                    mean_val = float(df[feature].mean())
                    std_val = float(df[feature].std())
                    input_data[feature] = st.number_input(
                        f"Enter {feature}",
                        value=mean_val,
                        step=std_val/10,
                        help=f"Mean: {mean_val:.2f}, Std: {std_val:.2f}"
                    )

                if st.button("Predict"):
                    input_df = pd.DataFrame([input_data])
                    prediction = make_prediction(
                        model, input_df, scaler, label_encoder, device
                    )
                    st.success(f"Predicted Class: {prediction[0]}")

            with pred_tab2:
                # Batch prediction interface
                st.write("Upload a CSV file for batch prediction")
                
                # Provide template download
                template_df = pd.DataFrame(columns=feature_columns)
                template_csv = template_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Template CSV",
                    template_csv,
                    "prediction_template.csv",
                    "text/csv"
                )

                # File upload for batch prediction
                batch_file = st.file_uploader(
                    "Upload CSV for batch prediction",
                    type=["csv"],
                    key="batch_pred"
                )
                
                if batch_file:
                    batch_df = pd.read_csv(batch_file)
                    if set(feature_columns).issubset(batch_df.columns):
                        batch_predictions = make_prediction(
                            model, batch_df[feature_columns], scaler, label_encoder, device
                        )
                        
                        # Create results DataFrame
                        results_df = batch_df.copy()
                        results_df['Predicted_Class'] = batch_predictions
                        
                        st.write("Prediction Results:")
                        st.dataframe(results_df)
                        
                        # Download predictions
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Predictions",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                    else:
                        st.error("Uploaded file missing required features!")
                        st.write("Required features:", feature_columns)

            # Add model performance visualization
            st.subheader("📊 Model Performance")
            perf_col1, perf_col2 = st.columns(2)

            with perf_col1:
                # Training/Validation Loss
                fig_loss = px.line(
                    title='Training Progress - Loss',
                    labels={'index': 'Epoch', 'value': 'Loss'}
                )
                fig_loss.add_scatter(y=metrics['train_losses'], name='Training Loss')
                fig_loss.add_scatter(y=metrics['val_losses'], name='Validation Loss')
                st.plotly_chart(fig_loss, use_container_width=True)

            with perf_col2:
                # Training/Validation Accuracy
                fig_acc = px.line(
                    title='Training Progress - Accuracy',
                    labels={'index': 'Epoch', 'value': 'Accuracy'}
                )
                fig_acc.add_scatter(y=metrics['train_accuracies'], name='Training Acc')
                fig_acc.add_scatter(y=metrics['val_accuracies'], name='Validation Acc')
                st.plotly_chart(fig_acc, use_container_width=True)

            # Add model summary
            st.subheader("📝 Model Summary")
            st.code(str(model))
