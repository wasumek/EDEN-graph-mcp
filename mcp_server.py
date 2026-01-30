import sys
import json
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import os

# Configure logging to stderr so it doesn't interfere with stdout JSON-RPC
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eden_mcp')

# --- EDEN Core Logic (Ported/Imported from app.py & market_analyzer.py) ---
# We reimplement/wrap minimal logic here to avoid Flask dependencies issues in stdio context if possible, 
# but for now we will reuse the logic structure.

# Load Resources
def load_resources():
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    nltk.data.path.append(nltk_data_dir)
    
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        logger.error("NLTK data missing.")
        return None, None, None

    try:
        df = pd.read_excel('0_Longevity World.xlsx', sheet_name='Data cleaned')
        # Anonymize Organization Names (Use 'Who_less' category + Index)
        df['Organization'] = [f"{row['Who_less']} {i+1}" for i, row in df.iterrows()]
    except Exception:
        logger.error("Dataset missing.")
        return None, None, None
        
    return df, stop_words, lemmatizer

DF, STOP_WORDS, LEMMATIZER = load_resources()

def preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    if not LEMMATIZER:
        return text.lower()
    tokens = word_tokenize(text.lower())
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens if token.isalpha() and token not in STOP_WORDS]
    return ' '.join(tokens)

def run_network_analysis(idea_text):
    """Core logic for /map endpoint"""
    if DF is None:
        return {"error": "Server initialization failed (missing data/NLTK)."}
        
    local_df = DF.copy()
    processed_idea = preprocess(idea_text)
    
    # Check if exists (Simplified logic)
    existing_row = local_df[local_df['Who_less'] == 'Your Idea']
    if existing_row.empty:
        new_row = {'Organization': 'New Organization', 'Who': 'New Value', 'Who_less': 'Your Idea', 'What': idea_text}
        local_df.loc[len(local_df)] = new_row
    else:
        local_df.loc[existing_row.index[0], 'What'] = idea_text

    # NLP Pipeline
    vectorizer = TfidfVectorizer()
    try:
        doc_term_matrix = vectorizer.fit_transform(local_df['What'].fillna('').apply(preprocess))
    except ValueError:
        return {"error": "Not enough data."}
    
    nmf = NMF(n_components=8, random_state=42, max_iter=1000)
    nmf.fit(doc_term_matrix)
    org_topic_matrix = nmf.transform(doc_term_matrix)
    cosine_sim = cosine_similarity(org_topic_matrix)

    # Get Top Matches
    your_idx = local_df[local_df['Who_less'] == 'Your Idea'].index[0]
    sim_scores = list(enumerate(cosine_sim[your_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_MATCHES = []
    
    for other_idx, score in sim_scores:
        if other_idx == your_idx: continue
        if len(top_MATCHES) >= 5: break
        
        row = local_df.iloc[other_idx]
        top_MATCHES.append({
            "organization": row['Organization'],
            "stakeholder_type": row['Who_less'],
            "similarity_score": round(score * 100, 1)
        })
        
    return {
        "network_analysis": {
            "input_idea": idea_text,
            "top_aligned_stakeholders": top_MATCHES
        }
    }

# --- MCP Protocol Implementation (JSON-RPC over Stdio) ---

def handle_request(request):
    try:
        if 'method' not in request:
            return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": None}
        
        method = request['method']
        
        # 1. Initialize
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request['id'],
                "result": {
                    "protocolVersion": "0.1.0",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "eden-graph-mcp",
                        "version": "1.0.0"
                    }
                }
            }
            
        # 2. List Tools
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request['id'],
                "result": {
                    "tools": [
                        {
                            "name": "eden_map_network",
                            "description": "Analyze a health value proposition to find aligned ecosystem stakeholders.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "idea": {"type": "string", "description": "The value proposition text to analyze."}
                                },
                                "required": ["idea"]
                            }
                        }
                    ]
                }
            }
            
        # 3. Call Tool
        if method == "tools/call":
            params = request.get('params', {})
            tool_name = params.get('name')
            args = params.get('arguments', {})
            
            if tool_name == "eden_map_network":
                result = run_network_analysis(args.get('idea', ''))
                return {
                    "jsonrpc": "2.0",
                    "id": request['id'],
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }]
                    }
                }
            else:
                return {"jsonrpc": "2.0", "id": request['id'], "error": {"code": -32601, "message": "Tool not found"}}

        return {"jsonrpc": "2.0", "id": request['id'], "result": {}} # Default ack

    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return {"jsonrpc": "2.0", "id": request.get('id'), "error": {"code": -32000, "message": str(e)}}

def main():
    logger.info("EDEN MCP Server Started")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            break

if __name__ == "__main__":
    main()
