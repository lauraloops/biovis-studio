import os, sys
HERE = os.path.dirname(__file__)
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..")),        # works for Home.py
    os.path.abspath(os.path.join(HERE, "..", "..")),  # works for pages/*
]
for ROOT in CANDIDATES:
    if os.path.isdir(os.path.join(ROOT, "core")) and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
        break

import streamlit as st
import pandas as pd
import plotly.express as px
from core.viz import make_scatter

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Try to import the real AI agent
AI_AVAILABLE = False
try:
    from ai.agent import analyze
    AI_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning(f"⚠️ AI module not found: {e}")
    def analyze(state, question):
        return {
            "text": "**AI Module Not Available**\n\nThe AI layer could not be imported. This is a stub response.",
            "parsed": None
        }

st.header("6) AI Assistant")

# Check if we have the minimum required state (prep_meta from DR step)
if "prep_meta" not in st.session_state:
    st.warning("No visualization data found. Please complete the pipeline:\n1. Import data\n2. Preprocess\n3. Run Dimensionality Reduction\n4. Create a visualization")
    st.stop()

# Extract existing state from session
prep_meta = st.session_state["prep_meta"]
raw_df = st.session_state.get("raw_df", pd.DataFrame())
X = st.session_state.get("X", pd.DataFrame())

# Reconstruct or find the visualization figure
# Since we're not modifying page 4, we need to recreate the figure here
# This is acceptable for the AI Assistant page as it's showing "current state"

st.subheader("Current Visualization")

# Detect available DR methods from column names
available_dr_methods = []
dr_coordinate_map = {}

if "PC1" in prep_meta.columns and "PC2" in prep_meta.columns:
    available_dr_methods.append("PCA")
    dr_coordinate_map["PCA"] = ("PC1", "PC2")

if "UMAP1" in prep_meta.columns and "UMAP2" in prep_meta.columns:
    available_dr_methods.append("UMAP")
    dr_coordinate_map["UMAP"] = ("UMAP1", "UMAP2")

if "TSNE1" in prep_meta.columns and "TSNE2" in prep_meta.columns:
    available_dr_methods.append("t-SNE")
    dr_coordinate_map["t-SNE"] = ("TSNE1", "TSNE2")

# Default to first available method
selected_dr_method = available_dr_methods[0] if available_dr_methods else None

# Recreate the plot based on current session state
# Try to get the last used config if it exists in session, otherwise use defaults
color_by = st.session_state.get("last_color_by", None)
shape_by = st.session_state.get("last_shape_by", None) 
size_by = st.session_state.get("last_size_by", None)

# Allow user to quickly adjust visualization parameters here
with st.expander(" Visualization Controls", expanded=False):
    # DR method selector if multiple methods available
    if len(available_dr_methods) > 1:
        selected_dr_method = st.radio(
            "Dimensionality Reduction Method",
            available_dr_methods,
            horizontal=True,
            help="Choose which DR technique to visualize"
        )
    elif len(available_dr_methods) == 1:
        st.info(f"📊 Using {available_dr_methods[0]} coordinates")
    
    cols = list(prep_meta.columns)
    color_by = st.selectbox("Color by", [None] + cols, index=0, key="ai_color")
    shape_by = st.selectbox("Shape by", [None] + cols, index=0, key="ai_shape")
    size_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(prep_meta[c])]
    size_by = st.selectbox("Size by (numeric)", [None] + size_candidates, index=0, key="ai_size")

# Create scatter plot with selected DR method
if selected_dr_method and selected_dr_method in dr_coordinate_map:
    x_col, y_col = dr_coordinate_map[selected_dr_method]
    viz_fig = px.scatter(prep_meta, x=x_col, y=y_col, color=color_by, symbol=shape_by, size=size_by)
    viz_fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
else:
    # Fallback to make_scatter if no method detected
    viz_fig = make_scatter(prep_meta, color_by=color_by, shape_by=shape_by, size_by=size_by)

viz_config = {"color_by": color_by, "shape_by": shape_by, "size_by": size_by}


# Display layout
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(viz_fig, use_container_width=True)

with col2:
    st.subheader("Data Preview")
    st.dataframe(prep_meta.head(20), use_container_width=True, height=300)
    
    with st.expander("📊 Session Info"):
        st.write(f"**Dataset shape:** {raw_df.shape if not raw_df.empty else 'N/A'}")
        st.write(f"**Processed shape:** {X.shape if not X.empty else 'N/A'}")
        st.write(f"**Embedding shape:** {prep_meta.shape}")
        
        # Show AI availability
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if AI_AVAILABLE and api_key:
            st.success("✅ AI Layer: Ready")
        elif AI_AVAILABLE and not api_key:
            st.warning("⚠️ AI Layer: API key missing")
        else:
            st.error("❌ AI Layer: Not available")

st.divider()

# Build the state object conforming to the Track A/B interface contract
ai_state = {
    "raw_df": raw_df,
    "viz_fig": viz_fig,
    "viz_points_df": prep_meta,
    "viz_config": viz_config,
    "dr_method": st.session_state.get("dr_method", "Unknown"),
    "dr_params": st.session_state.get("dr_params", {}),
    "dr_metrics": st.session_state.get("dr_metrics", {})
}

st.subheader("🤖 Ask the AI")

user_question = st.text_input(
    "What should the AI focus on?", 
    placeholder="e.g., Identify outliers, explain clusters, suggest next analyses...",
    help="Ask the AI to analyze your visualization and data"
)

if st.button(" Analyze", type="primary"):
    if not user_question.strip():
        st.warning("Please enter a question or topic for the AI to focus on.")
    else:
        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if AI_AVAILABLE and not api_key:
            st.error("❌ **Missing API Key**\n\nTo use the AI Assistant, you need to set your Gemini API key:\n\n"
                    "1. Get your API key from: https://aistudio.google.com/app/apikey\n"
                    "2. Add it to the `.env` file in the project root:\n   ```\n   GEMINI_API_KEY=your_key_here\n   ```\n"
                    "3. Restart the Streamlit app")
            st.stop()
        
        with st.spinner("AI is analyzing your data..."):
            try:
                # Call the AI layer (Track B implementation)
                response = analyze(ai_state, user_question)
                
                st.success("✅ Analysis Complete")      
                
                # Display Structured Response (Track B uses 'parsed' not 'json')
                if "parsed" in response and response["parsed"] is not None:
                    with st.expander("📋 Structured Output", expanded=True):
                        parsed = response["parsed"]
                        
                        # Display as a nice formatted structure
                        if hasattr(parsed, 'summary'):
                            st.markdown(f"**Summary:** {parsed.summary}")
                        
                        if hasattr(parsed, 'clusters') and parsed.clusters:
                            st.markdown("**Clusters:**")
                            for i, cluster in enumerate(parsed.clusters):
                                st.write(f"- {cluster.label}: {cluster.size} points")
                                if cluster.dominant_metadata:
                                    st.json(cluster.dominant_metadata)
                        
                        if hasattr(parsed, 'outliers') and parsed.outliers:
                            st.markdown(f"**Outliers:** {len(parsed.outliers)} detected")
                            st.write(parsed.outliers[:10])  # Show first 10
                        
                        if hasattr(parsed, 'hypotheses') and parsed.hypotheses:
                            st.markdown("**Hypotheses:**")
                            for h in parsed.hypotheses:
                                st.write(f"- {h}")
                        
                        if hasattr(parsed, 'next_steps') and parsed.next_steps:
                            st.markdown("**Suggested Next Steps:**")
                            for step in parsed.next_steps:
                                st.write(f"- {step}")
                        
                        if hasattr(parsed, 'caveats') and parsed.caveats:
                            with st.expander("⚠️ Caveats"):
                                for caveat in parsed.caveats:
                                    st.write(f"- {caveat}")
                        
                        # Also show raw JSON for debugging
                        with st.expander("Raw JSON"):
                            st.json(parsed.model_dump() if hasattr(parsed, 'model_dump') else str(parsed))
                        
            except RuntimeError as e:
                if "Missing GEMINI_API_KEY" in str(e):
                    st.error(" **Missing API Key**\n\nTo use the AI Assistant, you need to set your Gemini API key:\n\n"
                            "1. Get your API key from: https://aistudio.google.com/app/apikey\n"
                            "2. Add it to the `.env` file in the project root:\n   ```\n   GEMINI_API_KEY=your_key_here\n   ```\n"
                            "3. Restart the Streamlit app")
                else:
                    st.error(f"❌ An error occurred: {e}")
                    st.exception(e)
            except Exception as e:
                # Check for invalid API key error from Google GenAI
                error_str = str(e)
                if "API key not valid" in error_str or "API_KEY_INVALID" in error_str or "INVALID_ARGUMENT" in error_str:
                    st.error(" **Invalid API Key**")
                else:
                    st.error(f" An error occurred during analysis: {e}")
                    with st.expander("See error details"):
                        st.exception(e)


