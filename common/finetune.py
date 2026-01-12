import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from config import EMBEDDING_MODEL_NAME

DATA_FILE = "training_data.json"
OUTPUT_PATH = "fine_tuned_model"

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run generate_training_data.py first.")
        return
        
    print(f"Loading base model: {EMBEDDING_MODEL_NAME}")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Loading training data...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {DATA_FILE}: {e}")
        return
        
    if not data:
        print("No training data found.")
        return
        
    print(f"Found {len(data)} training pairs.")
    
    train_examples = []
    for item in data:
        # Use simple InputExample with texts=[query, positive]
        # MultipleNegativesRankingLoss expects (query, positive, [negative])
        # If we provide only (query, positive), it uses in-batch negatives
        train_examples.append(InputExample(texts=[item['anchor'], item['positive']]))
        
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Define loss function
    # MultipleNegativesRankingLoss is great for (query, positive) pairs
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    print("Starting training...")
    # Train the model
    # Adjust epochs as needed, 3-5 is usually a good start for small datasets
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=True
    )
    
    print(f"Saving fine-tuned model to '{OUTPUT_PATH}'...")
    model.save(OUTPUT_PATH)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()
