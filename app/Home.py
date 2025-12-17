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

st.set_page_config(page_title="BioVis Studio", layout="wide")

if "selected_step" not in st.session_state:
    st.session_state.selected_step = "Import"

WORKFLOW_STEPS = {
    "Import": {
        "title": "1️⃣ Import & Profile",
        "header": "Data Ingestion & Quality Check",
        "desc": "The entry point for your analysis. Load raw data and understand its structure.",
        "features": [
            "**File Support**: Upload CSV, TSV, or Parquet files.",
            "**Large Data**: Chunked loading for files >1GB.",
            "**Profiling**: Automatic calculation of missing values, unique counts, and data types.",
            "**Preview**: Inspect raw data tables before processing."
        ],
        "image_file": "import.png",
        "image_placeholder": "📊 Data Table & Stats View"
    },
    "Prep": {
        "title": "2️⃣ Data Preparation",
        "header": "Cleaning & Normalization",
        "desc": "Refine your dataset for downstream analysis.",
        "features": [
            "**Filtering**: Exclude samples based on metadata (e.g., remove 'Control' group).",
            "**Normalization**: Apply Log2, Z-score, or MinMax scaling.",
            "**Imputation**: Handle missing values with mean/median strategies.",
            "**Metadata**: Designate columns as categorical factors vs. numerical features."
        ],
        "image_file": "prep.png",
        "image_placeholder": "🔧 Filter Widgets & Histograms"
    },
    "DimRed": {
        "title": "3️⃣ Dimensionality Reduction",
        "header": "PCA, UMAP & t-SNE",
        "desc": "Project high-dimensional data into 2D or 3D space.",
        "features": [
            "**PCA**: Principal Component Analysis with Scree plots.",
            "**UMAP**: Non-linear reduction preserving local structure.",
            "**t-SNE**: Classic visualization for cluster separation.",
            "**Parameters**: Tune perplexity, neighbors, and learning rates."
        ],
        "image_file": "dimred.png",
        "image_placeholder": "📉 2D Projection Scatter"
    },
    "Visualize": {
        "title": "4️⃣ Visualize Studio",
        "header": "Interactive Exploration",
        "desc": "The core lab for discovering insights.",
        "features": [
            "**Interactive Plots**: Zoom, pan, and hover over data points.",
            "**Clustering**: Run K-Means, DBSCAN, or Hierarchical clustering on the fly.",
            "**Overlays**: Add regression lines, confidence intervals, and density contours.",
            "**Customization**: Change palettes, marker sizes, and themes."
        ],
        "image_file": "visualize.png",
        "image_placeholder": "🎨 Scatter Plot with Clusters"
    },
    "Export": {
        "title": "5️⃣ Export & Share",
        "header": "Publication & Reproducibility",
        "desc": "Save your work and share findings.",
        "features": [
            "**High-Res Figures**: Export plots as PNG, SVG, or PDF (300+ DPI).",
            "**Notebooks**: Generate a Python Jupyter Notebook to reproduce your analysis.",
            "**Data**: Download the processed and cleaned dataset.",
            "**Specs**: Save visualization configuration JSONs."
        ],
        "image_file": "export.png",
        "image_placeholder": "💾 Download Buttons & Notebook Code"
    }
}

def render_workflow_schema():
    # Custom CSS to make buttons look like workflow boxes
    st.markdown("""
    <style>
    div[data-testid="column"] button {
        height: 120px;
        width: 100%;
        border: 1px solid #41444e;
        background-color: #262730;
        color: #fafafa;
        border-radius: 8px;
        transition: all 0.2s;
        white-space: pre-wrap;
    }
    div[data-testid="column"] button:hover {
        border-color: #4a90e2;
        transform: scale(1.02);
        background-color: #2b3b55;
    }
    div[data-testid="column"] button:active {
        background-color: #4a90e2;
        color: white;
    }
    .arrow-text {
        font-size: 24px;
        text-align: center;
        color: #666;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 120px;
    }
    /* Light mode overrides */
    @media (prefers-color-scheme: light) {
        div[data-testid="column"] button {
            background-color: #f0f2f6;
            border: 1px solid #dce4ef;
            color: #31333F;
        }
        div[data-testid="column"] button:hover {
            background-color: #e8f0fe;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧬 BioVis Studio")
    st.markdown("### Workflow Overview")
    st.write("Click a step below to learn more about its capabilities:")
    
    # Layout: 5 buttons with arrows in between
    c1, a1, c2, a2, c3, a3, c4, a4, c5 = st.columns([2, 0.3, 2, 0.3, 2, 0.3, 2, 0.3, 2])

    with c1:
        if st.button("1️⃣ Import\nLoad & Profile", key="btn_import"):
            st.session_state.selected_step = "Import"
    with a1:
        st.markdown('<div class="arrow-text">➜</div>', unsafe_allow_html=True)

    with c2:
        if st.button("2️⃣ Prep\nClean & Normalize", key="btn_prep"):
            st.session_state.selected_step = "Prep"
    with a2:
        st.markdown('<div class="arrow-text">➜</div>', unsafe_allow_html=True)

    with c3:
        if st.button("3️⃣ DimRed\nPCA / UMAP / t-SNE", key="btn_dimred"):
            st.session_state.selected_step = "DimRed"
    with a3:
        st.markdown('<div class="arrow-text">➜</div>', unsafe_allow_html=True)

    with c4:
        if st.button("4️⃣ Visualize\nExplore & Cluster", key="btn_viz"):
            st.session_state.selected_step = "Visualize"
    with a4:
        st.markdown('<div class="arrow-text">➜</div>', unsafe_allow_html=True)

    with c5:
        if st.button("5️⃣ Export\nReport & Share", key="btn_export"):
            st.session_state.selected_step = "Export"

    # Detail Container
    selected = st.session_state.selected_step
    if selected in WORKFLOW_STEPS:
        info = WORKFLOW_STEPS[selected]
        
        st.markdown("---")
        with st.container(border=True):
            st.subheader(f"{info['title']}: {info['header']}")
            
            col_desc, col_viz = st.columns([1.5, 1])
            
            with col_desc:
                st.markdown(f"#### 📖 Description")
                st.write(info['desc'])
                
                st.markdown(f"#### ⚡ Key Functionalities")
                for feature in info['features']:
                    st.markdown(f"- {feature}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"Go to {selected} Page ➜", type="primary"):
                    st.switch_page(f"pages/{list(WORKFLOW_STEPS.keys()).index(selected) + 1}_{selected if selected != 'DimRed' else 'DimReduction'}{'_and_Profile' if selected == 'Import' else ''}{'_Data_Preparation' if selected == 'Prep' else ''}{'_Studio' if selected == 'Visualize' else ''}{'_and_Share' if selected == 'Export' else ''}.py")

            with col_viz:
                # Check if image exists in app/images/
                img_path = os.path.join(HERE, "images", info.get("image_file", ""))
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown(f"#### {info['image_placeholder']}")
                    # Placeholder for a real screenshot/plot
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #1e1e1e; 
                            border: 2px dashed #444; 
                            border-radius: 8px; 
                            height: 250px; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            color: #888;
                            text-align: center;
                            padding: 20px;
                        ">
                            (Add {info.get('image_file')} to app/images/)
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

    st.divider()

render_workflow_schema()
