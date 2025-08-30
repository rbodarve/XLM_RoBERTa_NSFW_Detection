#!/usr/bin/env python3
"""
Fine-tune DistilXLM-RoBERTa for NSFW word detection and export to ONNX/TFLite
Author: AI Assistant
Description: Complete pipeline for training, evaluating, and exporting NSFW detection model
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import tempfile
import subprocess
import sys

# ============================================================================
# 1. IMPORT LIBRARIES AND SETUP
# ============================================================================

def install_requirements():
    """Install required packages if not available"""
    required_packages = [
        'transformers[torch]',
        'datasets',
        'torch',
        'pandas',
        'scikit-learn',
        'onnx',
        'onnxruntime',
        'tensorflow',
        'optimum[onnxruntime]'
    ]
    
    for package in required_packages:
        try:
            __import__(package.split('[')[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages are installed
print("Checking and installing dependencies...")
install_requirements()

# Import additional libraries after installation
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import OptimizationConfig

print("All dependencies loaded successfully!")

# ============================================================================
# 2. LOAD AND PREPARE DATASET
# ============================================================================

def load_dataset(csv_path='dataset.csv'):
    """Load and validate the NSFW dataset"""
    print(f"Loading dataset from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Validate required columns
        required_cols = ['text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean and validate data
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(int)
        
        # Validate labels are binary (0 or 1)
        unique_labels = df['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            raise ValueError("Labels must be 0 (safe) or 1 (nsfw)")
        
        print(f"Dataset validation complete. Clean shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: Dataset file '{csv_path}' not found!")
        print("Creating a sample dataset for demonstration...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset for demonstration purposes"""
    sample_data = {
        'text': [
            # Safe words/phrases
            'hello', 'world', 'computer', 'programming', 'science', 'education',
            'family', 'friendship', 'learning', 'knowledge', 'book', 'music',
            'art', 'nature', 'technology', 'innovation', 'creativity', 'peace',
            'happiness', 'success', 'achievement', 'progress', 'development',
            'community', 'cooperation', 'collaboration', 'respect', 'kindness',
            # NSFW words (examples - replace with actual dataset)
            'inappropriate1', 'inappropriate2', 'inappropriate3', 'inappropriate4',
            'inappropriate5', 'inappropriate6', 'inappropriate7', 'inappropriate8',
            'inappropriate9', 'inappropriate10', 'inappropriate11', 'inappropriate12',
            'inappropriate13', 'inappropriate14', 'inappropriate15', 'inappropriate16',
            'inappropriate17', 'inappropriate18', 'inappropriate19', 'inappropriate20'
        ],
        'label': [0] * 28 + [1] * 20  # 28 safe, 20 nsfw
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('dataset.csv', index=False)
    print("Sample dataset created and saved as 'dataset.csv'")
    return df

def split_dataset(df, test_size=0.2, random_state=42):
    """Split dataset into training and validation sets"""
    print(f"Splitting dataset: {test_size*100}% for validation...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    return X_train, X_val, y_train, y_val

# ============================================================================
# 3. TOKENIZATION AND PREPROCESSING
# ============================================================================

def create_tokenized_datasets(X_train, X_val, y_train, y_val, model_name):
    """Tokenize the datasets using the model tokenizer"""
    print("Loading tokenizer and creating tokenized datasets...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'labels': y_train
    })
    
    val_dataset = Dataset.from_dict({
        'text': X_val,
        'labels': y_val
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Will be handled by data collator
            max_length=128  # Reasonable for word/short phrase detection
        )
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    print("Tokenization complete!")
    return train_tokenized, val_tokenized, tokenizer

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

def train_model(train_dataset, val_dataset, tokenizer, model_name, output_dir):
    """Train the DistilXLM-RoBERTa model"""
    print("Initializing model for training...")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "SAFE", 1: "NSFW"},
        label2id={"SAFE": 0, "NSFW": 1}
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # Disable wandb/tensorboard
        seed=42,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=True
    )
    
    def compute_metrics(eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the best model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer, model

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

def evaluate_model(trainer, X_val, y_val, output_dir):
    """Evaluate the trained model and save metrics"""
    print("Evaluating model performance...")
    
    # Get predictions
    eval_results = trainer.evaluate()
    
    # Get detailed predictions for classification report
    predictions = trainer.predict(trainer.eval_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Generate classification report
    report = classification_report(
        y_val, 
        y_pred, 
        target_names=['SAFE', 'NSFW'],
        digits=4
    )
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_val, y_pred)
    
    # Prepare metrics text
    metrics_text = f"""NSFW Detection Model Evaluation Results
{'='*50}

Accuracy: {accuracy:.4f}

Classification Report:
{report}

Training Results:
{'-'*30}
"""
    
    for key, value in eval_results.items():
        metrics_text += f"{key}: {value:.4f}\n"
    
    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    metrics_path = 'metrics/metrics.txt'
    
    with open(metrics_path, 'w') as f:
        f.write(metrics_text)
    
    print(f"Metrics saved to {metrics_path}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    return accuracy, report

# ============================================================================
# 6. ONNX EXPORT
# ============================================================================

def export_to_onnx(model_dir, onnx_path):
    """Export the trained model to ONNX format"""
    print("Exporting model to ONNX format...")
    
    try:
        # Load the trained model
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Create ONNX model
        onnx_model = ORTModelForSequenceClassification.from_pretrained(
            model_dir, 
            export=True
        )
        
        # Save ONNX model
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        onnx_model.save_pretrained(os.path.dirname(onnx_path))
        
        # Rename to desired filename
        onnx_source = os.path.join(os.path.dirname(onnx_path), "model.onnx")
        if os.path.exists(onnx_source):
            os.rename(onnx_source, onnx_path)
        
        print(f"ONNX model exported to: {onnx_path}")
        return True
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Attempting alternative ONNX export method...")
        
        try:
            # Alternative method using torch.onnx
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Create dummy input
            dummy_input = tokenizer(
                "sample text", 
                return_tensors="pt", 
                max_length=128, 
                padding="max_length", 
                truncation=True
            )
            
            # Export to ONNX
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size'},
                    'attention_mask': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            print(f"ONNX model exported to: {onnx_path}")
            return True
            
        except Exception as e2:
            print(f"Alternative ONNX export also failed: {e2}")
            return False

# ============================================================================
# 7. TENSORFLOW LITE EXPORT
# ============================================================================

def export_to_tflite(onnx_path, tflite_path):
    """Convert ONNX model to TensorFlow Lite"""
    print("Converting ONNX model to TensorFlow Lite...")
    
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert ONNX to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Create a temporary directory for SavedModel
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_model_path = os.path.join(temp_dir, "saved_model")
            tf_rep.export_graph(saved_model_path)
            
            # Convert SavedModel to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert
            tflite_model = converter.convert()
            
            # Save TFLite model
            os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite model exported to: {tflite_path}")
            return True
            
    except ImportError as e:
        print(f"TFLite export requires additional packages: {e}")
        print("Attempting alternative conversion method...")
        return export_tflite_alternative(onnx_path, tflite_path)
        
    except Exception as e:
        print(f"TFLite export failed: {e}")
        print("Attempting alternative conversion method...")
        return export_tflite_alternative(onnx_path, tflite_path)

def export_tflite_alternative(onnx_path, tflite_path):
    """Alternative TFLite export method using TensorFlow directly"""
    try:
        import tensorflow as tf
        from transformers import TFAutoModelForSequenceClassification
        
        # Load the PyTorch model and convert to TensorFlow
        model_dir = os.path.dirname(onnx_path).replace('/models', '/models/distilxlm_roberta_nsfw')
        
        # Load as TensorFlow model
        tf_model = TFAutoModelForSequenceClassification.from_pretrained(
            model_dir, 
            from_tf=False
        )
        
        # Create a concrete function for conversion
        @tf.function
        def representative_dataset():
            # Create dummy data for quantization
            input_ids = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
            attention_mask = tf.constant([[1, 1, 1, 1, 1]], dtype=tf.int32)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model exported to: {tflite_path}")
        return True
        
    except Exception as e:
        print(f"Alternative TFLite export failed: {e}")
        print("TFLite export skipped. ONNX model is available for inference.")
        return False

# ============================================================================
# 8. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training and export pipeline"""
    print("Starting NSFW Detection Model Training Pipeline")
    print("="*60)
    
    # Configuration
    MODEL_NAME = "distilbert/distilxlm-roberta-base"  # DistilXLM-RoBERTa model
    OUTPUT_DIR = "models/distilxlm_roberta_nsfw"
    ONNX_PATH = "models/nsfw_model.onnx"
    TFLITE_PATH = "models/nsfw_model.tflite"
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # Step 1: Load dataset
    df = load_dataset()
    
    # Step 2: Split dataset
    X_train, X_val, y_train, y_val = split_dataset(df)
    
    # Step 3: Create tokenized datasets
    train_dataset, val_dataset, tokenizer = create_tokenized_datasets(
        X_train, X_val, y_train, y_val, MODEL_NAME
    )
    
    # Step 4: Train model
    trainer, model = train_model(
        train_dataset, val_dataset, tokenizer, MODEL_NAME, OUTPUT_DIR
    )
    
    # Step 5: Evaluate model
    accuracy, report = evaluate_model(trainer, X_val, y_val, OUTPUT_DIR)
    
    # Step 6: Export to ONNX
    onnx_success = export_to_onnx(OUTPUT_DIR, ONNX_PATH)
    
    # Step 7: Export to TensorFlow Lite
    tflite_success = False
    if onnx_success:
        tflite_success = export_to_tflite(ONNX_PATH, TFLITE_PATH)
    
    # Final output
    print("\n" + "="*60)
    print("Export complete!")
    
    if onnx_success:
        print(f"ONNX model: {ONNX_PATH}")
    else:
        print("ONNX export: FAILED")
    
    if tflite_success:
        print(f"TFLite model: {TFLITE_PATH}")
    else:
        print("TFLite export: FAILED (ONNX model available)")
    
    print(f"\nModel checkpoints: {OUTPUT_DIR}")
    print(f"Metrics: metrics/metrics.txt")
    print(f"Final validation accuracy: {accuracy:.4f}")

# ============================================================================
# 9. INFERENCE EXAMPLE (BONUS)
# ============================================================================

def test_inference(model_dir, test_texts=None):
    """Test the trained model with sample texts"""
    if test_texts is None:
        test_texts = ["hello world", "inappropriate example"]
    
    print("\nTesting trained model...")
    
    try:
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        model.eval()
        
        for text in test_texts:
            # Tokenize
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            label = "SAFE" if predicted_class == 0 else "NSFW"
            print(f"Text: '{text}' -> {label} (confidence: {confidence:.4f})")
    
    except Exception as e:
        print(f"Inference test failed: {e}")

# ============================================================================
# 10. PROGRAM EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Run the main pipeline
        main()
        
        # Optional: Test inference
        test_inference("models/distilxlm_roberta_nsfw")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nProgram execution completed.")
