import math, time, os, argparse, json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Hyperparams ------------------
CONTEXT_LENGTH = 256
EMBED_DIM      = 256
FFN_DIM        = 512
DROPOUT        = 0.3

BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 50
STRIDE         = 16
WARMUP_EPOCHS  = 5
EARLY_STOP_PATIENCE = 10      # Early stopping

# ------------------ Dataset ------------------
class PTBDataset(Dataset):
    """Word-level PTB dataset with overlapping sliding window"""
    def __init__(self, path, vocab=None, seq_len=256, stride=None):
        self.seq_len = seq_len
        # Use stride to control overlap: stride=seq_len means no overlap
        # stride=1 means maximum overlap (standard sliding window)
        # stride=seq_len//2 means 50% overlap (good balance)
        self.stride = stride if stride is not None else seq_len // 2
        
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            text = text.replace("<unk>", "<UNK>")
            tokens = text.split()
        
        if vocab is None:
            unique_tokens = set(tokens)
            self.vocab = {"<unk>": 0}
            for token in sorted(unique_tokens):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        else:
            self.vocab = vocab
        
        self.ids = [self.vocab.get(t, 0) for t in tokens]

    def __len__(self):
        # Calculate number of samples based on stride
        if len(self.ids) <= self.seq_len:
            return 1
        return (len(self.ids) - self.seq_len) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        seq = self.ids[start: start + self.seq_len + 1]
        
        # Pad if necessary
        if len(seq) < self.seq_len + 1:
            seq = seq + [0] * (self.seq_len + 1 - len(seq))
        
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:],  dtype=torch.long)
        return x, y

# ------------------ Model ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, ffn_dim, dropout, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        # Weight tying
        self.fc_out.weight = self.embedding.weight

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        B, T = x.shape
        mask = self.generate_square_subsequent_mask(T).to(x.device)
        
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        logits = self.fc_out(x)
        return logits

# ------------------ Utils ------------------
def check_ptb_data():
    """Check if PTB data files exist"""
    required_files = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]
    for filename in required_files:
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filepath}\nPlease ensure all PTB data files are in the 'data/' folder.")
    print("✓ All PTB data files found")

@torch.no_grad()
def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def create_dataloaders(batch_size=BATCH_SIZE):
    """Create train, valid, test dataloaders"""
    # Use stride for overlapping windows
    train_dataset = PTBDataset("data/ptb.train.txt", seq_len=CONTEXT_LENGTH, stride=STRIDE)
    vocab = train_dataset.vocab
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train samples: {len(train_dataset)} (stride={STRIDE})")
    
    # For validation/test, use larger stride for faster evaluation
    valid_dataset = PTBDataset("data/ptb.valid.txt", vocab=vocab, seq_len=CONTEXT_LENGTH, stride=CONTEXT_LENGTH)
    test_dataset = PTBDataset("data/ptb.test.txt", vocab=vocab, seq_len=CONTEXT_LENGTH, stride=CONTEXT_LENGTH)
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return vocab, train_loader, valid_loader, test_loader

# ------------------ Training ------------------
def train_model(n_heads, n_layers, num_epochs=NUM_EPOCHS):
    """Train a single model configuration"""
    print(f"\n{'='*60}")
    print(f"Training: n_heads={n_heads}, n_layers={n_layers}")
    print(f"{'='*60}")
    
    check_ptb_data()
    vocab, train_loader, valid_loader, test_loader = create_dataloaders()
    
    model = TransformerLM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        max_len=CONTEXT_LENGTH
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Label smoothing to prevent overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # Learning rate scheduler with warmup and cosine decay
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (num_epochs - WARMUP_EPOCHS)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    history = {"epochs": [], "train_ppl": [], "valid_ppl": [], "test_ppl": []}
    best_valid_ppl = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_start = time.time()
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            train_ppl = calculate_perplexity(model, train_loader, device)
            valid_ppl = calculate_perplexity(model, valid_loader, device)
            test_ppl = calculate_perplexity(model, test_loader, device)
            
            history["epochs"].append(epoch + 1)
            history["train_ppl"].append(train_ppl)
            history["valid_ppl"].append(valid_ppl)
            history["test_ppl"].append(test_ppl)
            
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:2d} | Train PPL: {train_ppl:6.2f} | "
                  f"Valid PPL: {valid_ppl:6.2f} | Test PPL: {test_ppl:6.2f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Save best model and check early stopping
            if valid_ppl < best_valid_ppl:
                best_valid_ppl = valid_ppl
                patience_counter = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "n_heads": n_heads,
                    "n_layers": n_layers
                }, f"checkpoints/model_h{n_heads}_l{n_layers}.pt")
                print(f"  → New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE // 5:  # Divide by 5 since we eval every 5 epochs
                    print(f"  → Early stopping triggered (no improvement for {patience_counter * 5} epochs)")
                    break
        else:
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:2d} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    
    # Load best model and get final results
    checkpoint = torch.load(f"checkpoints/model_h{n_heads}_l{n_layers}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    final_train_ppl = calculate_perplexity(model, train_loader, device)
    final_valid_ppl = calculate_perplexity(model, valid_loader, device)
    final_test_ppl = calculate_perplexity(model, test_loader, device)
    
    total_time = time.time() - start_time
    
    # Plot training curves
    plot_training_curves(history, n_heads, n_layers)
    
    print(f"\nFinal Results:")
    print(f"Train PPL: {final_train_ppl:.2f}")
    print(f"Valid PPL: {final_valid_ppl:.2f}")
    print(f"Test PPL: {final_test_ppl:.2f}")
    print(f"Training time: {total_time:.1f}s")
    
    return {
        "n_heads": n_heads,
        "n_layers": n_layers,
        "train_ppl": final_train_ppl,
        "valid_ppl": final_valid_ppl,
        "test_ppl": final_test_ppl,
        "training_time": total_time
    }

def plot_training_curves(history, n_heads, n_layers):
    """Plot training and validation perplexity curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history["epochs"], history["train_ppl"], label="Train PPL", marker="o")
    plt.plot(history["epochs"], history["valid_ppl"], label="Valid PPL", marker="s")
    plt.plot(history["epochs"], history["test_ppl"], label="Test PPL", marker="^")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title(f"Training Curves (Heads={n_heads}, Layers={n_layers})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/training_h{n_heads}_l{n_layers}.png", dpi=150)
    plt.close()

# ------------------ Main Functions ------------------
def run_single_experiment(n_heads, n_layers):
    """Run a single experiment with specified configuration"""
    result = train_model(n_heads, n_layers)
    
    # Save individual result
    os.makedirs("results", exist_ok=True)
    with open(f"results/result_h{n_heads}_l{n_layers}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResult saved to results/result_h{n_heads}_l{n_layers}.json")
    return result

def run_batch_experiments(head_configs, layer_configs):
    """Run batch experiments with multiple configurations"""
    results = []
    
    for n_heads in head_configs:
        for n_layers in layer_configs:
            result = train_model(n_heads, n_layers)
            results.append(result)
    
    # Create results table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Heads':<8} {'Layers':<8} {'Train PPL':<12} {'Valid PPL':<12} {'Test PPL':<12} {'Time (s)':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['n_heads']:<8} {r['n_layers']:<8} {r['train_ppl']:<12.2f} "
              f"{r['valid_ppl']:<12.2f} {r['test_ppl']:<12.2f} {r['training_time']:<10.1f}")
    
    print("="*80)
    
    # Save all results to file
    os.makedirs("results", exist_ok=True)
    with open("results/all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll results saved to results/all_results.json")
    
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Transformer Language Model on PTB dataset')
    
    # Model configuration
    parser.add_argument('--heads', type=int, nargs='+',
                       help='Number of attention heads. Can be a single value or multiple values for batch experiments')
    parser.add_argument('--layers', type=int, nargs='+',
                       help='Number of transformer layers. Can be a single value or multiple values for batch experiments')
    
    # Training configuration (optional)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--embed-dim', type=int, default=EMBED_DIM,
                       help=f'Embedding dimension (default: {EMBED_DIM})')
    parser.add_argument('--ffn-dim', type=int, default=FFN_DIM,
                       help=f'FFN hidden dimension (default: {FFN_DIM})')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                       help=f'Dropout rate (default: {DROPOUT})')
    parser.add_argument('--context-length', type=int, default=CONTEXT_LENGTH,
                       help=f'Context length (default: {CONTEXT_LENGTH})')
    parser.add_argument('--stride', type=int, default=STRIDE,
                       help=f'Stride for sliding window (default: {STRIDE})')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'all'], default='single',
                       help='Training mode: single (one config), batch (multiple configs), or all (default 3x3 grid)')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Update global variables if provided
    global NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, EMBED_DIM, FFN_DIM, DROPOUT, CONTEXT_LENGTH, STRIDE
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EMBED_DIM = args.embed_dim
    FFN_DIM = args.ffn_dim
    DROPOUT = args.dropout
    CONTEXT_LENGTH = args.context_length
    STRIDE = args.stride
    
    if args.mode == 'all':
        # Run default 3x3 grid experiments
        print("Running all 9 configurations (3x3 grid)...")
        head_configs = [2, 4, 8]
        layer_configs = [2, 4, 8]
        run_batch_experiments(head_configs, layer_configs)
        
    elif args.mode == 'batch':
        # Run batch experiments with specified configurations
        if not args.heads or not args.layers:
            print("Error: --heads and --layers must be specified for batch mode")
            print("Example: python script.py --mode batch --heads 2 4 8 --layers 2 4 8")
            return
        
        print(f"Running batch experiments with heads={args.heads} and layers={args.layers}")
        run_batch_experiments(args.heads, args.layers)
        
    else:  # single mode (default)
        # Run single experiment
        if not args.heads or not args.layers:
            print("Error: --heads and --layers must be specified for single mode")
            print("Example: python script.py --heads 4 --layers 6")
            print("Or use --mode all to run all default configurations")
            return
        
        if len(args.heads) > 1 or len(args.layers) > 1:
            print("Warning: Multiple values provided. Use --mode batch for multiple configurations.")
            print("Using first values only...")
        
        n_heads = args.heads[0]
        n_layers = args.layers[0]
        
        # Check if embed_dim is divisible by n_heads
        if EMBED_DIM % n_heads != 0:
            print(f"Error: embed_dim ({EMBED_DIM}) must be divisible by n_heads ({n_heads})")
            print(f"Try using n_heads from: {[i for i in [1,2,4,8,16,32] if EMBED_DIM % i == 0]}")
            return
        
        print(f"Running single experiment with {n_heads} heads and {n_layers} layers")
        run_single_experiment(n_heads, n_layers)

if __name__ == "__main__":
    main()
