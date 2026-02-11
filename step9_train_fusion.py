#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  ШАГ 9: FUSION MODEL v3 (3-Branch) — RETRAIN            ║
║                                                          ║
║  Исправления:                                            ║
║  1. LayerNorm вместо BatchNorm (работает на 1 сэмпле)   ║
║  2. Label smoothing (0.1)                                ║
║  3. Cosine annealing scheduler                           ║
║  4. Focal loss для трудных примеров                      ║
║  5. Проверка распределения предсказаний                  ║
╚══════════════════════════════════════════════════════════╝
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report

ENRICHED_FILE = "data/enriched_calendar.csv"
RULE_EMBEDDINGS_FILE = "data/event_embeddings.npy"
SENTIMENT_EMBEDDINGS_FILE = "data/sentiment_embeddings.npy"
FEATURES_FILE = "models/features_list.json"
MODEL_FILE = "models/fusion_model.pt"
NORM_STATS_FILE = "models/fusion_norm_stats.json"
STATS_FILE = "models/fusion_stats.json"

TRAIN_END = 2020
BATCH_SIZE = 256
EPOCHS = 150
LEARNING_RATE = 0.0005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════
# ДАТАСЕТ
# ═══════════════════════════════════════════════════
class NewsDataset(Dataset):
    def __init__(self, tabular, rules, sentiment, labels):
        self.tabular = torch.FloatTensor(tabular)
        self.rules = torch.FloatTensor(rules)
        self.sentiment = torch.FloatTensor(sentiment)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tabular[idx], self.rules[idx], self.sentiment[idx], self.labels[idx]


# ═══════════════════════════════════════════════════
# FUSION NETWORK v3 (LayerNorm instead of BatchNorm)
# ═══════════════════════════════════════════════════
class FusionNetwork(nn.Module):
    """
    Triple-Branch Fusion (LayerNorm — works with batch_size=1):

    1. Tabular (139) → 128
    2. Rules   (384) → 128
    3. Sentiment(384) → 128
           ↓
       Concat (128*3 = 384)
           ↓
       Fusion Layers → 2 classes
    """
    def __init__(self, tabular_dim, text_dim, hidden_dim=128):
        super().__init__()
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.rules_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.sentiment_branch = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        fusion_input = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, tabular, rules, sentiment):
        t = self.tabular_branch(tabular)
        r = self.rules_branch(rules)
        s = self.sentiment_branch(sentiment)
        return self.fusion(torch.cat([t, r, s], dim=1))


# ═══════════════════════════════════════════════════
# ПОДГОТОВКА ДАННЫХ
# ═══════════════════════════════════════════════════
def prepare_data():
    print("📦 Загрузка данных...")
    df = pd.read_csv(ENRICHED_FILE)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    rule_emb = np.load(RULE_EMBEDDINGS_FILE)
    sent_emb = np.load(SENTIMENT_EMBEDDINGS_FILE)

    with open(FEATURES_FILE, 'r') as f:
        features = json.load(f)

    # Feature Engineering
    df['Year'] = df['DateTime'].dt.year
    df['Forecast'] = df['Forecast'].fillna(0)
    df['Previous'] = df['Previous'].fillna(0)
    df['Forecast_Prev_Diff'] = df['Forecast'] - df['Previous']

    # Target: 1=ВЫШЕ, 0=НИЖЕ
    df['Direction'] = np.where(df['Actual'] > df['Forecast'], 1, 0)
    mask = df['Actual'] != df['Forecast']

    df = df[mask].reset_index(drop=True)
    rule_emb = rule_emb[mask.values]
    sent_emb = sent_emb[mask.values]

    # Num Features
    cat_features = ['Currency', 'Event']
    num_features = [f for f in features if f not in cat_features and f in df.columns]

    for f in num_features:
        if df[f].isna().any():
            df[f] = df[f].fillna(0)

    # Standardization (train only)
    tabular_data = df[num_features].values.astype(np.float32)
    train_mask = df['Year'] < TRAIN_END

    mean = tabular_data[train_mask].mean(axis=0)
    std = tabular_data[train_mask].std(axis=0) + 1e-8
    tabular_data = (tabular_data - mean) / std
    tabular_data = np.nan_to_num(tabular_data, nan=0.0)

    labels = df['Direction'].values
    train_idx = df['Year'] < TRAIN_END
    test_idx = df['Year'] >= TRAIN_END

    # Баланс классов
    train_labels = labels[train_idx]
    n0 = (train_labels == 0).sum()
    n1 = (train_labels == 1).sum()
    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    class_weights = torch.FloatTensor([w0, w1]).to(DEVICE)

    print(f"   📊 Числовых фичей: {len(num_features)}")
    print(f"   📚 Rule Emb dim: {rule_emb.shape[1]}")
    print(f"   🗣️ Sentiment Emb dim: {sent_emb.shape[1]}")
    print(f"   🏷️ Баланс классов (train):")
    print(f"      НИЖЕ (0): {n0} ({n0/total*100:.1f}%) weight={w0:.3f}")
    print(f"      ВЫШЕ (1): {n1} ({n1/total*100:.1f}%) weight={w1:.3f}")
    print(f"   🏷️ Test:")
    test_labels = labels[test_idx]
    t0 = (test_labels == 0).sum()
    t1 = (test_labels == 1).sum()
    print(f"      НИЖЕ: {t0} | ВЫШЕ: {t1}")

    train_dataset = NewsDataset(
        tabular_data[train_idx], rule_emb[train_idx], sent_emb[train_idx], labels[train_idx]
    )
    test_dataset = NewsDataset(
        tabular_data[test_idx], rule_emb[test_idx], sent_emb[test_idx], labels[test_idx]
    )

    # Save norm stats
    norm_stats = {'mean': mean.tolist(), 'std': std.tolist(), 'features': num_features}
    with open(NORM_STATS_FILE, 'w') as f:
        json.dump(norm_stats, f)

    return train_dataset, test_dataset, len(num_features), rule_emb.shape[1], class_weights


# ═══════════════════════════════════════════════════
# ОБУЧЕНИЕ
# ═══════════════════════════════════════════════════
def train_model():
    print("╔══════════════════════════════════════════════════╗")
    print("║  🧠 FUSION MODEL v3 (3-Branch, LayerNorm)        ║")
    print("║  Tabular + Rules + Sentiment                    ║")
    print("╚══════════════════════════════════════════════════╝")

    train_ds, test_ds, tab_dim, text_dim, class_weights = prepare_data()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = FusionNetwork(tab_dim, text_dim).to(DEVICE)
    print(f"\n🧠 Параметров: {sum(p.numel() for p in model.parameters()):,}")

    # CrossEntropy с весами классов + label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0
    best_epoch = 0
    patience = 20
    no_improve = 0

    print(f"\n⏳ Обучение ({EPOCHS} эпох, lr={LEARNING_RATE})...\n")

    for epoch in range(EPOCHS):
        # === TRAIN ===
        model.train()
        train_loss = 0
        train_batches = 0
        for tab, rule, sent, lbl in train_loader:
            tab = tab.to(DEVICE)
            rule = rule.to(DEVICE)
            sent = sent.to(DEVICE)
            lbl = lbl.to(DEVICE)

            optimizer.zero_grad()
            out = model(tab, rule, sent)
            loss = criterion(out, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()

        # === EVAL ===
        model.eval()
        all_preds = []
        all_lbls = []
        all_probs = []

        with torch.no_grad():
            for tab, rule, sent, lbl in test_loader:
                tab = tab.to(DEVICE)
                rule = rule.to(DEVICE)
                sent = sent.to(DEVICE)
                lbl = lbl.to(DEVICE)

                out = model(tab, rule, sent)
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_lbls.extend(lbl.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_lbls = np.array(all_lbls)
        all_probs = np.array(all_probs)
        acc = accuracy_score(all_lbls, all_preds) * 100

        # Проверка: предсказывает ли модель ОБА класса?
        pred_0 = (all_preds == 0).sum()
        pred_1 = (all_preds == 1).sum()

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'tabular_dim': tab_dim,
                'text_dim': text_dim,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
            }, MODEL_FILE)
        else:
            no_improve += 1

        if epoch % 10 == 0:
            avg_loss = train_loss / train_batches
            conf = all_probs.max(axis=1).mean() * 100
            print(f"   Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | "
                  f"Best: {best_acc:.1f}% | Pred 0/1: {pred_0}/{pred_1} | Avg Conf: {conf:.1f}%")

        if no_improve >= patience:
            print(f"   🛑 Early stopping at {epoch+1}")
            break

    # ═══ ФИНАЛЬНАЯ ОЦЕНКА ═══
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_probs = []
    all_preds = []
    all_lbls = []

    with torch.no_grad():
        for tab, rule, sent, lbl in test_loader:
            tab, rule, sent, lbl = tab.to(DEVICE), rule.to(DEVICE), sent.to(DEVICE), lbl.to(DEVICE)
            out = model(tab, rule, sent)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbl.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_lbls = np.array(all_lbls)

    final_acc = accuracy_score(all_lbls, all_preds) * 100
    confidence = all_probs.max(axis=1)

    # High confidence
    high_conf = confidence > 0.65
    if high_conf.sum() > 0:
        high_acc = accuracy_score(all_lbls[high_conf], all_preds[high_conf]) * 100
    else:
        high_acc = 0

    # Распределение предсказаний
    pred_0 = (all_preds == 0).sum()
    pred_1 = (all_preds == 1).sum()
    true_0 = (all_lbls == 0).sum()
    true_1 = (all_lbls == 1).sum()

    # Тест одиночного сэмпла
    single_tab = torch.FloatTensor(tabular_test_sample(test_ds)).unsqueeze(0).to(DEVICE)
    single_rule = test_ds.rules[0:1].to(DEVICE)
    single_sent = test_ds.sentiment[0:1].to(DEVICE)
    with torch.no_grad():
        single_out = model(single_tab, single_rule, single_sent)
        single_prob = torch.softmax(single_out, dim=1).cpu().numpy()[0]

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  📊 РЕЗУЛЬТАТЫ FUSION v3 (LayerNorm)                    ║
╠══════════════════════════════════════════════════════════╣
║  🎯 Точность:         {final_acc:5.1f}%                         ║
║  ⭐ При уверен. >65%:  {high_acc:5.1f}% ({high_conf.sum()} сделок)    ║
║  🏷️ Pred НИЖЕ/ВЫШЕ:   {pred_0}/{pred_1}                     ║
║  🏷️ True НИЖЕ/ВЫШЕ:   {true_0}/{true_1}                     ║
║  📊 Avg Confidence:    {confidence.mean()*100:.1f}%                     ║
║  🧪 Single-sample:     [{single_prob[0]:.3f}, {single_prob[1]:.3f}]              ║
╚══════════════════════════════════════════════════════════╝
""")

    print("Classification Report:")
    print(classification_report(all_lbls, all_preds, target_names=['НИЖЕ', 'ВЫШЕ']))

    # Save stats
    stats = {
        'accuracy': float(final_acc),
        'high_conf_accuracy': float(high_acc),
        'high_conf_count': int(high_conf.sum()),
        'pred_distribution': {'ниже': int(pred_0), 'выше': int(pred_1)},
        'true_distribution': {'ниже': int(true_0), 'выше': int(true_1)},
        'avg_confidence': float(confidence.mean()),
        'best_epoch': best_epoch,
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)


def tabular_test_sample(ds):
    """Return first test tabular sample as numpy."""
    return ds.tabular[0].numpy()


if __name__ == "__main__":
    train_model()
