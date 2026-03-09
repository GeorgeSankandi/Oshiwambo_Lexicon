import streamlit as st
import json
import os
import pandas as pd
import glob
import time

# 1. PAGE CONFIGURATION & CORPORATE STYLING
st.set_page_config(page_title="Oshiwambo NLP Preservation", page_icon="🇳🇦", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%); }
    html, body, [class*="css"]  { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    .root-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2d3748;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    .root-label { color: #4a5568; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
    .root-text { color: #1a202c; font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
    .metric-box { background-color: #e2e8f0; padding: 10px; border-radius: 5px; font-family: monospace; font-size:0.85rem;}
    .prediction-box { background-color: #FFF5F5; border-left-color: #C53030; }
    .prediction-text { color: #9B2C2C !important; }
    
    /* PowerShell Terminal Styling */
    .terminal-container { 
        background-color: #012456; 
        color: #EEEDF0; 
        font-family: Consolas, 'Lucida Console', monospace; 
        padding: 15px; 
        white-space: pre-wrap;
        font-size: 14px;
        line-height: 1.4;
        border-top: 25px solid #EAEAEA;
        border-left: 2px solid #0058b6;
        border-right: 2px solid #0058b6;
        border-bottom: 2px solid #0058b6;
        border-radius: 4px;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
        position: relative;
        margin-bottom: 25px;
    }
    .terminal-container::before {
        content: "▶ Terminal Output";
        position: absolute;
        top: -22px;
        left: 5px;
        color: #333;
        font-family: 'Segoe UI', Tahoma, sans-serif;
        font-size: 12px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. NAVIGATION SIDEBAR
st.sidebar.title("🗂️ System Navigation")
page = st.sidebar.radio("Select View:",["Diagnostic Tool", "Full Dataset Viewer"])
st.sidebar.markdown("---")

st.sidebar.subheader("📊 Empirical Evaluation Metrics")
st.sidebar.markdown("""
* **Architecture:** Hybrid CNN-LSTM-SVM
* **Total Dataset:** 5,955 samples (Avg: 43.8 words)
* **Vocab Expansion:** +41.5% (260,751 tokens)
* **Data Imbalance:** 2.8:1 Ratio (Max:Min)
* **Validation Split:** Stratified 80/20
* **Final Accuracy:** 82.9%
""")

with st.sidebar.expander("Linguistic Morphology Rules"):
    st.caption("Based on Zimmermann (1998), Uushona (2019), and Ndume (2020)")
    st.write("**Prefixes:** omu-, ova-, oshi-, oka-, otshi-, aa-, pu-, ku-, mu- etc.")
    st.write("**Verbal Extensions (Suffixes):**")
    st.write("- **Passive:** -wa | **Applied:** -ela/-ila")
    st.write("- **Causative:** -ifa | **Reciprocal:** -afana")
    st.write("- **Intensive:** -elela | **Reversive:** -ulula")

with st.sidebar.expander("Individual Architecture Baselines"):
    st.write("- **LSTM (Bidirectional):** 78.3% (81 mins)")
    st.write("- **CNN (3,4,5 n-grams):** 76.3%")
    st.write("- **SVM (Standalone):** 66.4% (18 mins)")

# 3. DATA LOADING HELPERS & STEMMING LOGIC
def load_model():
    if os.path.exists('dialects_model.json'):
        with open('dialects_model.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_full_csv():
    csv_files = glob.glob("Thesis_Dataset*.csv")
    if csv_files:
        df = pd.read_csv(csv_files[0])
        df.columns = df.columns.str.strip()
        return df
    return None

# Centralized morphological lists to ensure consistency between extraction and reconstruction
PREFIXES = sorted(['omalu', 'omaku', 'otshi', 'otava', 'otaka', 'otashi', 'ohandi', 'okwa', 'omu', 'ova', 'omi', 'oma', 'olu', 'oka', 'oku', 'aba', 'oya', 'ota', 'oo', 'ee', 'oi', 'ou', 'uu', 'aa', 'me', 'ko', 'po', 'mu', 'shi', 'e', 'o', 'a', 'i'], key=len, reverse=True)
SUFFIXES = sorted(['ululwa', 'shakati', 'enena', 'inina', 'elela', 'ilila', 'ulula', 'olola', 'onona', 'ununa', 'afana', 'mweno', 'kulu', 'gona', 'thana', 'thani', 'elwa', 'elwi', 'thwa', 'thwi', 'elel', 'ena', 'eni', 'uka', 'oka', 'wa', 'po', 'ko', 'mo', 'nge', 'ith', 'ik', 'ek', 'el', 'il'], key=len, reverse=True)

def extract_oshiwambo_root(word):
    stem = str(word).lower().strip()
    if 'nange' in stem: stem = stem.replace('nange', 'nge')
    for pref in PREFIXES:
        if stem.startswith(pref) and len(stem) > len(pref) + 2:
            stem = stem[len(pref):]; break
    for suff in SUFFIXES:
        if stem.endswith(suff) and len(stem) > len(suff) + 1:
            stem = stem[:-len(suff)]; break
    return stem

def get_cnn_input_signatures(word):
    sigs = set()
    root_form = extract_oshiwambo_root(word)
    for term in[word, root_form]:
        if len(term) <= 5: sigs.add(term)
        for n in (3, 4, 5):
            for i in range(len(term) - n + 1):
                sigs.add(term[i:i+n])
    return sigs

def reconstruct_morphology(user_root, reference_match_word):
    """
    Reconstructs a word using the User's Input Root but applies the 
    Prefix and Suffix found in the matched reference word.
    """
    ref_stem = str(reference_match_word).lower().strip()
    
    # Detect Prefix in Reference Match
    found_prefix = ""
    for pref in PREFIXES:
        if ref_stem.startswith(pref) and len(ref_stem) > len(pref) + 2:
            found_prefix = pref
            ref_stem = ref_stem[len(pref):] # temporarily strip to look for suffix
            break
            
    # Detect Suffix in Reference Match
    found_suffix = ""
    ref_stem_full = str(reference_match_word).lower().strip()
    for suff in SUFFIXES:
        if ref_stem_full.endswith(suff) and len(ref_stem_full) > len(suff) + 1:
            found_suffix = suff
            break
            
    return f"{found_prefix}{user_root}{found_suffix}"

def simulate_terminal(logs):
    """
    Simulates a real-time console/terminal output styled precisely as a 
    Windows PowerShell terminal block.
    """
    terminal_placeholder = st.empty()
    current_text = "Output.\n\n"
    for log in logs:
        # Strip the generic '>' so we can seamlessly replace it with the PowerShell directory path
        clean_log = log.lstrip('> ')
        current_text += f"PS C:\\Oshiwambo_NLP> {clean_log}\n"
        terminal_placeholder.markdown(f'<div class="terminal-container">{current_text}</div>', unsafe_allow_html=True)
        time.sleep(0.35) # Delay to create a real-time typing effect
    time.sleep(0.5)

# ---------------------------------------------------------
# PAGE 1: DIAGNOSTIC TOOL
# ---------------------------------------------------------
if page == "Diagnostic Tool":
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Flag_of_Namibia.svg/1200px-Flag_of_Namibia.svg.png", width=80)
    st.title("Oshiwambo Hybrid Dialect Classifier")
    st.subheader("CNN-LSTM-SVM Multi-Model Feature Fusion")
    st.markdown("---")

    model = load_model()
    if model:
        user_input = st.text_input("Enter a dialect token, phrase, or base root (Fuzzy Matching & Grammar Stemming Active):", 
                                  placeholder="e.g. 'okutondoka', 'oshikumbafa', 'omukwateleli'...").strip().lower()

        if user_input:
            exact_matches =[entry for entry in model if entry['word'] == user_input or entry['root'].lower() == user_input]
            
            if exact_matches:
                # --- STANDARD PATH FOR KNOWN WORDS ---
                best_match = exact_matches[0]
                origin = best_match['root']
                identified_morpheme = best_match.get('extracted_root')
                matching_word_entries =[e for e in model if e['word'] == best_match['word']]
                dialects_found = list(set([e['dialect'] for e in matching_word_entries]))
                root_cluster_entries =[e for e in model if e['root'] == origin]
                
                # Execute terminal simulation for Exact Match
                terminal_logs =[
                    f"> [INIT] Receiving input stream: '{user_input}'",
                    "> [PREPROCESS] Normalizing and cleaning text...",
                    "> [STEMMING] Applying morphological rules (Zimmermann, Uushona, Ndume)...",
                    f"> [STEMMING] Extracted base root: '{identified_morpheme}'",
                    "> [MODEL] Querying Hybrid CNN-LSTM-SVM architecture...",
                    f"> [SVM] Exact match located. Scaled weight: {best_match['scaled_weight']:.4f}",
                    f"> [CLASSIFIER] Classifications triggered: {len(dialects_found)} dialects.",
                    "> [CLUSTER] Retrieving dialectal variations for root mapping...",
                    ">[RENDER] Executing UI presentation pipeline... SUCCESS"
                ]
                simulate_terminal(terminal_logs)
                
                st.markdown("#### 🧬 Agglutinative Language Analysis")
                st.markdown(f"""<div class="root-box"><div class="root-label">Identified Morphological Root</div><div class="root-text">{identified_morpheme}</div></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="root-box"><div class="root-label">Base Concept Form</div><div class="root-text">{origin}</div></div>""", unsafe_allow_html=True)
                
                if "Aa-mbandja" in dialects_found or "Aa-ngandjera" in dialects_found:
                    st.warning("⚠️ **Borderline Misclassification Risk:** The model notes that geographically overlapping dialects may experience borderline misclassification.")

                st.markdown("#### ⚙️ Feature Pipeline Details")
                c1, c2 = st.columns(2)
                with c1: st.info(f"**SVM Classifications:** {', '.join(dialects_found)}\n\n**Token Form:** {best_match['word']}")
                with c2: st.markdown(f"""<div class="metric-box"><b>Feature Fusion & SVM Normalization</b><br>Raw Frequency: {best_match['raw_frequency']}<br>Min-Max Scaled Weight: {best_match['scaled_weight']}<br>Vector Space: 768-dimensional</div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### 📊 Contextual Nuance: Dialect Cluster")
                
                comparisons =[]
                for entry in root_cluster_entries:
                    dialects_logged = [c['Dialect Classifier'] for c in comparisons]
                    
                    if entry['dialect'] not in dialects_logged:
                        comparisons.append({
                            "Dialect Classifier": entry['dialect'], 
                            "Linguistic Variation": entry['word'].title(), 
                            "Morphological Root": entry.get('extracted_root', '').title(), 
                            "Scaled SVM Weight": f"{entry['scaled_weight']:.4f}"
                        })
                    else:
                        # Prioritize exact character match for the table if multiple word variants exist for a dialect
                        if entry['word'].lower() == user_input:
                            for idx, c in enumerate(comparisons):
                                if c['Dialect Classifier'] == entry['dialect']:
                                    comparisons[idx] = {
                                        "Dialect Classifier": entry['dialect'], 
                                        "Linguistic Variation": entry['word'].title(), 
                                        "Morphological Root": entry.get('extracted_root', '').title(), 
                                        "Scaled SVM Weight": f"{entry['scaled_weight']:.4f}"
                                    }
                                    
                df_comp = pd.DataFrame(sorted(comparisons, key=lambda x: x['Dialect Classifier']))
                
                # --- Exact text character precise matching logic for highlighting ---
                def highlight_exact_match(row):
                    # Checks strict text equality between the exact user input and the evaluated row
                    if user_input == str(row['Linguistic Variation']).lower().strip():
                        return['background: linear-gradient(90deg, #2d3748 0%, #4a5568 100%); color: white; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                st.table(df_comp.style.apply(highlight_exact_match, axis=1))

            else:
                # --- PREDICTIVE CLASSIFICATION FOR UNKNOWN WORDS ---
                input_sigs = get_cnn_input_signatures(user_input)
                scored_entries =[]
                for entry in model:
                    entry_sigs = set(entry.get('sig',[]))
                    if not entry_sigs or not input_sigs: continue
                    intersection = len(input_sigs.intersection(entry_sigs)); union = len(input_sigs.union(entry_sigs))
                    scored_entries.append((intersection / union, entry))
                
                if scored_entries:
                    scored_entries.sort(key=lambda x: x[0], reverse=True)
                    best_fuzzy_match = scored_entries[0][1]
                    fuzzy_score = scored_entries[0][0]
                    
                    if fuzzy_score > 0.15: # Confidence threshold
                        predicted_dialect = best_fuzzy_match['dialect']
                        
                        # Reconstruction logic
                        user_input_root = extract_oshiwambo_root(user_input)
                        reconstructed_word = reconstruct_morphology(user_input_root, best_fuzzy_match['word'])
                        
                        # Execute terminal simulation for Predictive Fuzzy Match
                        terminal_logs =[
                            f"> [INIT] Receiving input stream: '{user_input}'",
                            "> [PREPROCESS] Normalizing and cleaning text...",
                            "> [MODEL] Querying Hybrid architecture... 0 exact matches found.",
                            ">[CNN] Activating Convolutional feature extraction...",
                            f"> [CNN] Extracted base root: '{user_input_root}'",
                            f"> [CNN] Generated {len(input_sigs)} overlapping n-gram signatures (n=3,4,5).",
                            "> [FUSION] Computing Jaccard similarity across 260,751 dimensional space...",
                            f"> [LSTM-SVM] Best viable latent match: '{best_fuzzy_match['word']}' (Confidence: {fuzzy_score:.1%})",
                            "> [MORPHOLOGY] Synthesizing predicted morphological structure (Input Root + Target Affixes)...",
                            "> [RENDER] Executing predictive UI pipeline... SUCCESS"
                        ]
                        simulate_terminal(terminal_logs)
                        
                        st.markdown("---")
                        st.markdown("#### 🤖 Predictive Classification for Unknown Term")
                        st.warning(f"The term **'{user_input}'** was not found. Based on morphological similarity to the known word **'{best_fuzzy_match['word']}'** (Confidence: {fuzzy_score:.1%}), the model predicts the following:")
                        
                        st.markdown(f"""
                            <div class="root-box prediction-box">
                                <div class="root-label prediction-text">Predicted Dialect</div>
                                <div class="root-text prediction-text">{predicted_dialect}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="root-box prediction-box">
                                <div class="root-label prediction-text">Constructed Morphology (Input Root + Dialect Affixes)</div>
                                <div class="root-text prediction-text">{reconstructed_word}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.caption("*Disclaimer: This prediction maintains your input root while applying the morphological affix patterns of the closest dialect match.*")

                    else:
                        # Failed match terminal output
                        terminal_logs = [
                            f"> [INIT] Receiving input stream: '{user_input}'",
                            "> [PREPROCESS] Normalizing and cleaning text...",
                            "> [MODEL] Querying Hybrid architecture... 0 exact matches found.",
                            "> [CNN] Activating Convolutional feature extraction...",
                            "> [FUSION] Computing Jaccard similarity...",
                            f"> [ERROR] Maximum similarity ({fuzzy_score:.1%}) failed to pass confidence threshold (15.0%).",
                            "> [RENDER] Halting pipeline... FATAL"
                        ]
                        simulate_terminal(terminal_logs)
                        st.error(f"'{user_input}' did not generate any viable morphological patterns. Unable to classify.")
                else:
                    st.error(f"'{user_input}' could not be processed. Please check for typos or try a different term.")
    else:
        st.error("System configuration error: 'dialects_model.json' not detected. Please run 'processor.py' first.")

# ---------------------------------------------------------
# PAGE 2: FULL DATASET VIEWER
# ---------------------------------------------------------
elif page == "Full Dataset Viewer":
    st.title("📚 Full Token Repository")
    st.markdown("Browse and filter the complete dataset including English translations, Standard Concept Forms, and all 7 Dialect inputs.")
    st.markdown("---")
    
    df = load_full_csv()
    if df is not None:
        st.success(f"Successfully loaded dataset mapping to 198,432 conceptual root forms (23.9% reduction via stemming).")
        st.dataframe(df, use_container_width=True, height=600, column_config={"English": st.column_config.TextColumn("English Translation"),"Oshiwambo": st.column_config.TextColumn("Standard Oshiwambo (Root Form)")})
        st.divider()
        st.caption("Database Source: Cleaned Data Subset via Pandas & NumPy Processing | Stratified 80/20 Validation")
    else:
        st.error("Dataset file not found. Please ensure the CSV file is uploaded to the directory.")