from flask import Flask, request, render_template, jsonify
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

# Check if nltk_data exists locally and append to path
if os.path.exists('./nltk_data'):
    nltk.data.path.append('./nltk_data')
    
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import networkx as nx
import numpy as np
import openai
from contextlib import redirect_stdout
import plotly.graph_objects as go
from dotenv import load_dotenv
from market_analyzer import MarketAnalyzer

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='static')

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
file_path = '0_Longevity World.xlsx'
sheet_name = 'Data cleaned'
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Anonymize Organization Names (Use 'Who_less' category + Index)
    df['Organization'] = [f"{row['Who_less']} {i+1}" for i, row in df.iterrows()]
except Exception as e:
    print(f"Warning: Could not load dataset {file_path}. Error: {e}")
    df = pd.DataFrame()

# Initialize NLP tools
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    print("Warning: NLTK data not found. Please run setup_nltk.py")
    stop_words = set()
    lemmatizer = None

def preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    if not lemmatizer:
        return text.lower()
        
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get form data from request
        form_data = request.form
        print("Received form data:", form_data)
        
        # Validate input data
        try:
            founded_year = int(form_data['founded_year'])
            investors = int(form_data['investors'])
            funding = float(form_data['funding'])
            employees = int(form_data['employees'])
        except ValueError as e:
            return jsonify({'error': 'Invalid input data format. Please check your numbers.'}), 400
        
        # Check if file exists
        if not os.path.exists('DHT_T2D_market_analysis_cleaned.xlsx'):
            return jsonify({'error': 'Analysis file not found'}), 404
            
        analyzer = MarketAnalyzer('DHT_T2D_market_analysis_cleaned.xlsx')
        
        # Pass form data to collect_user_data
        analyzer.collect_user_data(
            market_segment=form_data['market_segment'],
            product=form_data['product'],
            customer=form_data['customer'],
            founded_year=founded_year,
            investors=investors,
            funding=funding,
            employees=employees
        )
        
        analyzer.perform_clustering()
        
        # Generate and convert plots with error checking
        try:
            dendrogram = analyzer.plot_dendrogram()
            overview = analyzer.plot_cluster_overview()
            comparison = analyzer.plot_cluster_comparison()
            
            result = {
                'dendrogram': dendrogram.to_json() if dendrogram else None,
                'overview': overview.to_json() if overview else None,
                'comparison': comparison.to_json() if comparison else None
            }
            
            return jsonify(result)
            
        except Exception as plot_error:
            print(f"Error generating plots: {str(plot_error)}")
            return jsonify({'error': 'Error generating analysis plots'}), 500
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    new_idea_text = data.get('What', '')
    prompt = f"Who is the customer, what customer problems does the company solve, and what needs does the company meet? Based on this data: {new_idea_text}"

    if not api_key:
         return "OpenAI API Key not found. Please set it in .env file."

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI that helps analyze value propositions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

@app.route('/map', methods=['POST'])
def map_network():
    data = request.get_json()
    new_idea_text = preprocess(data.get('What', ''))

    local_df = df.copy()

    # Check if the new idea already exists in the DataFrame
    # Using 'Who_less' as a unique identifier for "Your Idea"
    existing_row = local_df[local_df['Who_less'] == 'Your Idea']

    if existing_row.empty:
        # If it doesn't exist, add a new row
        new_row = {
            'Organization': 'New Organization',
            'Who': 'New Value',
            'Who_less': 'Your Idea',
            'What': new_idea_text
        }
        local_df.loc[len(local_df)] = new_row
    else:
        # If it exists, update the existing row
        local_df.loc[existing_row.index[0], 'What'] = new_idea_text

    # Process text data
    value_propositions = local_df['What'].fillna('')
    preprocessed_vp = value_propositions.apply(preprocess)

    # Create a document-term matrix
    vectorizer = TfidfVectorizer()
    try:
        doc_term_matrix = vectorizer.fit_transform(preprocessed_vp)
    except ValueError:
        return jsonify({'error': 'Not enough data to perform analysis.'}), 400

    # Apply the LDA model (NMF in this case)
    nmf = NMF(n_components=8, random_state=42, max_iter=1000)
    nmf.fit(doc_term_matrix)
    
    # Calculate cosine similarities between organization topic representations
    org_topic_matrix = nmf.transform(doc_term_matrix)
    cosine_sim = cosine_similarity(org_topic_matrix)

    # Map organizations to their segments
    segments = local_df['Who_less']
    segment_dict = defaultdict(list)
    for i, seg in enumerate(segments):
        segment_dict[seg].append(i)

    # Calculate average cosine similarity between segments
    segment_sim = {}
    for seg1 in segment_dict:
        for seg2 in segment_dict:
            if seg1 < seg2:  # Only once per pair
                indices1 = segment_dict[seg1]
                indices2 = segment_dict[seg2]
                similarities = [cosine_sim[i][j] for i in indices1 for j in indices2]
                avg_similarity = np.mean(similarities) if similarities else 0
                segment_sim[(seg1, seg2)] = avg_similarity

    # Flatten and sort similarities to find the 50th percentile cutoff
    similarity_values = list(segment_sim.values())
    cutoff = np.percentile(similarity_values, 50) if similarity_values else 0

    # Create a graph with segments as nodes
    G = nx.Graph()

    # Add nodes for each unique segment
    for seg in segment_dict:
        G.add_node(seg)

    # Add edges between segments if similarity is above the cutoff
    for (seg1, seg2), weight in segment_sim.items():
        if weight >= cutoff:
            G.add_edge(seg1, seg2, weight=weight)

    # Get positions for the nodes
    pos = nx.spring_layout(G)

    # Create Plotly graph
    edge_traces = []
    edge_annotations = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=edge[2]['weight'] * 10, color='#888'),
            hoverinfo='text',
            mode='lines',
            text=[f'{edge[2]["weight"]:.2f}'])
        edge_traces.append(edge_trace)
        
    node_x = []
    node_y = []
    node_degree = []
    node_size = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = len(list(G.neighbors(node)))
        node_degree.append(degree)
        node_text.append(node)
        
        if node == 'Your Idea':
            node_size.append(25)
        else:
            node_size.append(10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_degree,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='Stakeholder Alignment Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    graph_data = fig.to_dict()

    # Recommendations Logic
    # 1. Vectorize new idea
    # 2. Find most similar existing segments
    
    # Re-vectorize specific new idea string to get distance to others
    # (Simplified approach: reuse existing cosine matrix logic or similarity computation)
    
    # Find similarity of "Your Idea" to others
    your_idea_idx = local_df[local_df['Who_less'] == 'Your Idea'].index
    
    similar_segments_output = "### MOST ALIGNED STAKEHOLDERS\n"
    collaboration_suggestions = ""

    if not your_idea_idx.empty:
        idx = your_idea_idx[0]
        # Get similarities for this specific doc index
        # org_topic_matrix is (n_samples, n_topics)
        # cosine_sim is (n_samples, n_samples)
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Top 3 similar, excluding itself
        top_3 = [x for x in sim_scores if x[0] != idx][:3]
        
        for rank, (other_idx, score) in enumerate(top_3, 1):
             org_name = local_df.iloc[other_idx]['Organization']
             who_less = local_df.iloc[other_idx]['Who_less']
             similarity_pct = score * 100
             similar_segments_output += f"**Rank {rank}**: {who_less} ({org_name}) - {similarity_pct:.1f}%\n"

        # AI Recommendations
        if api_key:
             collaboration_prompt = (
                "Based on the identified similar organizations, provide 3 succinct bullet points on how to collaborate. "
                "Focus on: 1. Population reach. 2. User retention. 3. Scalable payment."
            )
             try:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a research assistant for digital health ecosystems."},
                        {"role": "user", "content": f"{collaboration_prompt} My Idea: {new_idea_text}. Aligned Stakeholders: {similar_segments_output}"}
                    ],
                    max_tokens=300
                )
                collaboration_suggestions = response.choices[0].message.content.strip()
             except openai.AuthenticationError:
                collaboration_suggestions = "AI Suggestions unavailable: Invalid API Key."
             except openai.APIConnectionError:
                collaboration_suggestions = "AI Suggestions unavailable: Connection failed."
             except openai.RateLimitError:
                collaboration_suggestions = "AI Suggestions unavailable: Rate limit exceeded or billing inactive."
             except Exception as e:
                print(f"OpenAI Error: {str(e)}") # Log the full error
                collaboration_suggestions = "AI Suggestions unavailable (Check console for details)."
    
    return jsonify(graph_data=graph_data, suggestions=similar_segments_output + "\n\n### STRATEGIC RECOMMENDATIONS\n" + collaboration_suggestions)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
