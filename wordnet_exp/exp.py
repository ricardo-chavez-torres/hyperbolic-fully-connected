import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from sklearn.metrics import f1_score
import sys
from pathlib import Path
import random

# --- Setup Paths ---
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from layers import Lorentz_fully_connected, Lorentz

# --- Constants ---
SEED = 42
EMBEDDING_DIM = 5      
LR = 0.005             # Increased slightly for stability with the scaler
EPOCHS = 100
BATCH_SIZE = 32

# The datasets used in HNN++ (Shimizu et al.) / Ganea et al.
# We run a separate experiment for each subtree.
DATASETS = [
    'animal.n.01',
    'group.n.01',
    'mammal.n.01',
    'location.n.01',
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- 1. Dynamic Data Preparation ---
def load_subtree_data(root_name):
    """
    Generates a classification task for a specific subtree.
    The CLASSES are the immediate children (hyponyms) of the root.
    The INPUTS are all descendants of those children.
    """
    print(f"\n📚 Loading Subtree: {root_name}...")
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet', quiet=True)

    root_syn = wn.synset(root_name)
    
    # Target Classes = The immediate children (siblings)
    # e.g., if root is 'mammal', classes are 'carnivore', 'rodent', etc.
    target_synsets = root_syn.hyponyms()
    
    # Filter out tiny classes (optional, but helps stability)
    # We keep classes that have at least some depth/descendants
    target_synsets = [s for s in target_synsets if len(list(s.closure(lambda x: x.hyponyms()))) > 5]
    
    class_names = [s.name() for s in target_synsets]
    node_to_class = {}
    
    print(f"   Found {len(class_names)} sub-categories (classes).")

    # Assign descendants to classes
    for class_id, target_syn in enumerate(target_synsets):
        # Get all descendants
        descendants = set(s.name() for s in target_syn.closure(lambda s: s.hyponyms()))
        descendants.add(target_syn.name())
        
        for d_name in descendants:
            # First Claim Wins (Mutual Exclusivity)
            if d_name not in node_to_class:
                node_to_class[d_name] = class_id

    # Create Vocabulary
    sorted_vocab = sorted(list(node_to_class.keys()))
    word_to_idx = {w: i for i, w in enumerate(sorted_vocab)}
    
    # Tensors
    X = torch.tensor([word_to_idx[w] for w in sorted_vocab], dtype=torch.long)
    y = torch.tensor([node_to_class[w] for w in sorted_vocab], dtype=torch.long)
    
    print(f"   Vocabulary Size: {len(X)}")
    print(f"   Class Distribution: {torch.bincount(y).tolist()}")
    
    return X, y, len(X), len(class_names), class_names

# --- 2. The Model ---
class HyperbolicClassifier(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_classes, manifold, layer_type='ours'):
        super().__init__()
        self.manifold = manifold
        self.layer_type = layer_type
        
        # Embedding Layer
        self.embeddings = nn.Embedding(num_nodes, embedding_dim + 1)
        # self.init_embeddings()

        # Classifier
        if layer_type == 'euclidean':
            self.linear = nn.Linear(embedding_dim + 1, num_classes)
        elif layer_type == 'ours':
            self.linear = Lorentz_fully_connected(
                in_features=embedding_dim, 
                out_features=num_classes, 
                manifold=manifold,
                activation=nn.Identity(), 
                do_mlr=True
            )

    # def init_embeddings(self):
    #     # Initialize in tangent space near origin
    #     k = self.manifold.k()
    #     w = torch.randn(self.embeddings.num_embeddings, self.embeddings.embedding_dim - 1) * 0.01
        
    #     # ExpMap0
    #     w_norm = torch.norm(w, dim=-1, keepdim=True)
    #     w_norm = torch.clamp(w_norm, min=1e-8)
    #     sqrt_k = torch.sqrt(k)
        
    #     time = (1/sqrt_k) * torch.cosh(sqrt_k * w_norm)
    #     space = (1/sqrt_k) * (torch.sinh(sqrt_k * w_norm) / w_norm) * w
        
    #     with torch.no_grad():
    #         self.embeddings.weight.data[:, 0] = time.squeeze()
    #         self.embeddings.weight.data[:, 1:] = space

    def forward(self, idx):
        x = self.embeddings(idx)
        logits = self.linear(x)
        return logits

# --- 3. Training Loop ---
def train_and_evaluate(X, y, num_nodes, num_classes, class_names, manifold, layer_type):
    print(f"   Running {layer_type.upper()}...")
    
    model = HyperbolicClassifier(num_nodes, EMBEDDING_DIM, num_classes, manifold, layer_type).double().cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Split
    indices = torch.randperm(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    # Dataloaders
    train_ds = torch.utils.data.TensorDataset(X[train_idx], y[train_idx])
    test_ds = torch.utils.data.TensorDataset(X[test_idx], y[test_idx])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)

    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            
            # Gradient clipping is crucial for hyperbolic stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Projection
            if layer_type != 'euclidean':
                with torch.no_grad():
                    w = model.embeddings.weight.data
                    space = w[:, 1:]
                    k = manifold.k()
                    new_time = torch.sqrt(torch.norm(space, dim=1)**2 + 1/k)
                    model.embeddings.weight.data[:, 0] = new_time

    # Final Eval
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
            pred = model(batch_X).argmax(dim=1)
            all_preds.append(pred)
            all_targets.append(batch_y)
            
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    macro_f1 = f1_score(all_targets.cpu(), all_preds.cpu(), average='macro')
    print(f"   🏁 Final Macro-F1 ({layer_type}): {macro_f1:.4f}")
    return macro_f1

# --- Main Runner ---
def run_experiment():
    set_seed(SEED)
    manifold = Lorentz(k=1.0)
    
    print(f"🚀 Starting Subtree Experiments (Dims={EMBEDDING_DIM})")
    print("=" * 60)
    
    results = {}
    
    for dataset_name in DATASETS:
        # Load data specific to this subtree
        X, y, num_nodes, num_classes, class_names = load_subtree_data(dataset_name)
        
        # Skip if too small
        if num_classes < 2:
            print("   ⚠️  Skipping: Not enough subclasses.")
            continue
            
        print("-" * 40)
        
        # Train Euclidean
        f1_e = train_and_evaluate(X, y, num_nodes, num_classes, class_names, manifold, 'euclidean')
        
        # Train Ours
        f1_o = train_and_evaluate(X, y, num_nodes, num_classes, class_names, manifold, 'ours')
        
        results[dataset_name] = (f1_e, f1_o)

    # Summary Table
    print("\n" + "=" * 60)
    print("📊 SUBTREE CLASSIFICATION RESULTS (Macro F1)")
    print("=" * 60)
    print(f"{'Dataset':<20} | {'Euclidean':<10} | {'Ours':<10} | {'Delta':<10}")
    print("-" * 60)
    for name, (e, o) in results.items():
        delta = o - e
        print(f"{name:<20} | {e:.4f}     | {o:.4f}     | {delta:+.4f}")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()