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
from core.viz import make_scatter

# Try to import the AI agent, otherwise stub it
try:
    from core.ai.agent import analyze
except ImportError:
    def analyze(state, question):
        st.subheader("state")
        st.write(state)
        return {
            "text": f"**Stub AI Response**\n\nYou asked: *'{question}'*\n\nI can see you're working with:\n- Dataset: {state.get('raw_df', pd.DataFrame()).shape if isinstance(state.get('raw_df'), pd.DataFrame) else 'N/A'}\n- DR Method: {state.get('dr_method', 'Unknown')}\n- Visualization points: {state.get('viz_points_df', pd.DataFrame()).shape if isinstance(state.get('viz_points_df'), pd.DataFrame) else 'N/A'}\n\n*Note: This is a placeholder. The actual AI layer (Track B) will provide real insights.*",
            "json": {
                "clusters_summary": "Pending implementation",
                "outliers": [],
                "hypotheses": ["Requires Track B implementation"],
                "next_steps": ["Integrate Gemini LLM", "Build context builder"],
                "caveats": ["Stub response only"]
            }
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

# Recreate the plot based on current session state
# Try to get the last used config if it exists in session, otherwise use defaults
color_by = st.session_state.get("last_color_by", None)
shape_by = st.session_state.get("last_shape_by", None) 
size_by = st.session_state.get("last_size_by", None)

# Allow user to quickly adjust visualization parameters here
with st.expander("Visualization Controls", expanded=False):
    cols = list(prep_meta.columns)
    color_by = st.selectbox("Color by", [None] + cols, index=0, key="ai_color")
    shape_by = st.selectbox("Shape by", [None] + cols, index=0, key="ai_shape")
    size_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(prep_meta[c])]
    size_by = st.selectbox("Size by (numeric)", [None] + size_candidates, index=0, key="ai_size")

viz_fig = make_scatter(prep_meta, color_by=color_by, shape_by=shape_by, size_by=size_by)
viz_config = {"color_by": color_by, "shape_by": shape_by, "size_by": size_by}

# Display layout
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(viz_fig, use_container_width=True)

with col2:
    st.subheader("Data Preview")
    st.dataframe(prep_meta.head(20), use_container_width=True, height=300)
    
    with st.expander("Session Info"):
        st.write(f"**Dataset shape:** {raw_df.shape if not raw_df.empty else 'N/A'}")
        st.write(f"**Processed shape:** {X.shape if not X.empty else 'N/A'}")
        st.write(f"**Embedding shape:** {prep_meta.shape}")
        

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
        with st.spinner("AI is analyzing your data..."):
            try:
                # Call the AI layer (stubbed for now, Track B will implement)
                response = analyze(ai_state, user_question)
                
                st.success("Analysis Complete")
                
                # Display Text Response
                if "text" in response and response["text"]:
                    st.markdown("### Insights")
                    st.markdown(response["text"])
                
                # Display JSON/Structured Response
                if "json" in response and response["json"]:
                    with st.expander("Structured Output (JSON)"):
                        st.json(response["json"])
                        
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)
