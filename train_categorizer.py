import pandas as pd
import tensorflow as tf
import sentencepiece as spm
import os
import glob
import numpy as np
import time
import random

# ======================================================
# CONFIGURATION (Strictly from Methodology Text)
# ======================================================
VOCAB_SIZE = 800        # "Vocabulary size: 800 subword units"
SEQ_LENGTH = 32         # "Maximum sequence length: 32 tokens"
D_MODEL = 256           # "Model dimension: 256"
NUM_LAYERS = 4          # "Encoder/Decoder layers: 4"
NUM_HEADS = 4           # "Attention heads: 4"
DFF = 1024              # "Feed-forward dimension: 1024"
DROPOUT_RATE = 0.2      # "Dropout: 0.20"
BATCH_SIZE = 32         # "Batch size: 32"
EPOCHS = 50             # "Maximum epochs: 100" (Adjusted for demo speed)
LEARNING_RATE = 3e-4    # "Learning rate = 3e-4"
LABEL_SMOOTHING = 0.1   # "Label smoothing = 0.10"

# Special Tokens (ONLY Domain Specific)
CUSTOM_TOKENS = ['<CUR>', '<REV>']
DIALECT_TAGS = ['<DIALECT=NDONGA>', '<DIALECT=KWANYAMA>', '<DIALECT=MBALANTU>', 
                '<DIALECT=KWAMBI>', '<DIALECT=KWALUUDHI>', '<DIALECT=NGANDJERA>', '<DIALECT=MBANDJA>']

def train_transformer_system():
    print("=== STARTING OSHI-EXPAND (STAGE A) TRAINING ===")

    # 1. LOAD DATA & CONSTRUCT PAIRS
    csv_files = glob.glob("Thesis_Dataset*.csv")
    if not csv_files: print("❌ No CSV found."); return
    df = pd.read_csv(csv_files[0])
    df.columns = df.columns.str.strip()

    dialect_map = {
        'Aa-ndonga': '<DIALECT=NDONGA>', 'Aa-kwanyama': '<DIALECT=KWANYAMA>',
        'Aa-mbalanhu': '<DIALECT=MBALANTU>', 'Aa-kwambi': '<DIALECT=KWAMBI>',
        'Aa-kwaluudhi': '<DIALECT=KWALUUDHI>', 'Aa-ngandjera': '<DIALECT=NGANDJERA>',
        'Aa-mbandja': '<DIALECT=MBANDJA>'
    }

    grouped_data = {} 
    
    print("Constructing Multitask Training Pairs...")
    for _, row in df.iterrows():
        current_term = str(row.get('Oshiwambo', '')).strip()
        if not current_term or current_term.lower() == 'nan': continue
        
        if current_term not in grouped_data: grouped_data[current_term] = []

        for col, tag in dialect_map.items():
            dialect_term = str(row.get(col, '')).strip()
            if dialect_term and dialect_term.lower() != 'nan':
                src_fwd = f"{tag} <CUR> {current_term}"
                tgt_fwd = dialect_term
                grouped_data[current_term].append((src_fwd, tgt_fwd))
                
                src_rev = f"{tag} <REV> {dialect_term}"
                tgt_rev = current_term
                grouped_data[current_term].append((src_rev, tgt_rev))

    all_sentences = []
    for term, pairs in grouped_data.items():
        for src, tgt in pairs:
            all_sentences.append(src)
            all_sentences.append(tgt)
            
    with open('corpus.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_sentences))

    # 2. TRAIN TOKENIZER
    print(f"Training SentencePiece (Unigram, Vocab={VOCAB_SIZE}, Byte Fallback=True)...")
    
    user_defined = ",".join(CUSTOM_TOKENS + DIALECT_TAGS)
    
    spm.SentencePieceTrainer.train(
        input='corpus.txt', 
        model_prefix='oshi_spm', 
        vocab_size=VOCAB_SIZE,
        model_type='unigram', 
        character_coverage=1.0, 
        byte_fallback=True, 
        user_defined_symbols=user_defined,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        pad_piece='<pad>', bos_piece='<bos>', eos_piece='<eos>', unk_piece='<unk>'
    )
    
    sp = spm.SentencePieceProcessor(model_file='oshi_spm.model')

    def encode_text(text_list):
        ids = [sp.encode_as_ids(t) for t in text_list]
        return tf.keras.preprocessing.sequence.pad_sequences(
            [[1] + x + [2] for x in ids], maxlen=SEQ_LENGTH, padding='post'
        )

    # 3. SPLIT STRATEGY
    unique_terms = list(grouped_data.keys())
    random.shuffle(unique_terms)
    split_idx = int(len(unique_terms) * 0.8)
    
    train_terms = unique_terms[:split_idx]
    val_terms = unique_terms[split_idx:]
    
    def get_dataset(terms_list):
        src_data, tgt_data = [], []
        for t in terms_list:
            for s, tr in grouped_data[t]:
                src_data.append(s); tgt_data.append(tr)
        return encode_text(src_data), encode_text(tgt_data)

    X_train, y_train = get_dataset(train_terms)
    X_val, y_val = get_dataset(val_terms)
    
    print(f"Training Pairs: {len(X_train)} | Validation Pairs: {len(X_val)}")

    # 4. TRANSFORMER ARCHITECTURE
    def transformer_encoder(inputs):
        x = tf.keras.layers.Embedding(VOCAB_SIZE, D_MODEL)(inputs)
        pos = tf.keras.layers.Embedding(SEQ_LENGTH, D_MODEL)(tf.range(start=0, limit=SEQ_LENGTH, delta=1))
        x = x + pos
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

        for _ in range(NUM_LAYERS):
            attn = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL//NUM_HEADS)(x, x)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn) 
            
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(DFF, activation='gelu'),
                tf.keras.layers.Dropout(DROPOUT_RATE),
                tf.keras.layers.Dense(D_MODEL)
            ])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
        return x

    def transformer_decoder(inputs, enc_outputs):
        x = tf.keras.layers.Embedding(VOCAB_SIZE, D_MODEL)(inputs)
        pos = tf.keras.layers.Embedding(SEQ_LENGTH, D_MODEL)(tf.range(start=0, limit=SEQ_LENGTH, delta=1))
        x = x + pos
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

        for _ in range(NUM_LAYERS):
            attn1 = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL//NUM_HEADS)(x, x, use_causal_mask=True)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn1)
            
            attn2 = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=D_MODEL//NUM_HEADS)(x, enc_outputs)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn2)
            
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(DFF, activation='gelu'),
                tf.keras.layers.Dropout(DROPOUT_RATE),
                tf.keras.layers.Dense(D_MODEL)
            ])
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
        return x

    enc_inputs = tf.keras.Input(shape=(SEQ_LENGTH,), name='encoder_inputs')
    dec_inputs = tf.keras.Input(shape=(SEQ_LENGTH,), name='decoder_inputs')
    
    enc_outputs = transformer_encoder(enc_inputs)
    dec_outputs = transformer_decoder(dec_inputs, enc_outputs)
    
    final_outputs = tf.keras.layers.Dense(VOCAB_SIZE)(dec_outputs)
    
    model = tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=final_outputs, name="oshi_transformer")

    # 5. OPTIMIZATION
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01, clipnorm=1.0)
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    model.summary()

    # 6. TRAINING LOOP (FIXED TARGET SHAPES)
    # ---------------------------------------------------------
    print(f"Starting Training for {EPOCHS} Epochs...")
    
    # Slice inputs (remove EOS for input) and targets (remove BOS for target)
    y_train_inp = y_train[:, :-1]
    y_train_real = y_train[:, 1:]
    
    y_val_inp = y_val[:, :-1]
    y_val_real = y_val[:, 1:]
    
    # FIXED: Pad back to SEQ_LENGTH so dimensions match (32 vs 32)
    # This ensures (Batch, 32) aligns with model output (Batch, 32, Vocab)
    y_train_inp = tf.keras.preprocessing.sequence.pad_sequences(y_train_inp, maxlen=SEQ_LENGTH, padding='post')
    y_train_real = tf.keras.preprocessing.sequence.pad_sequences(y_train_real, maxlen=SEQ_LENGTH, padding='post')
    
    y_val_inp = tf.keras.preprocessing.sequence.pad_sequences(y_val_inp, maxlen=SEQ_LENGTH, padding='post')
    y_val_real = tf.keras.preprocessing.sequence.pad_sequences(y_val_real, maxlen=SEQ_LENGTH, padding='post')

    model.fit([X_train, y_train_inp], y_train_real, 
              validation_data=([X_val, y_val_inp], y_val_real),
              epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 7. SAVE WEIGHTS
    model.save_weights('transformer_weights.weights.h5')
    print("✅ Model weights saved.")
    print("🚀 TRAINING COMPLETE. Run app.py now.")

if __name__ == "__main__":
    train_transformer_system()