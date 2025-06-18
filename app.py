import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# 🎯 Advanced Page Configuration
st.set_page_config(
    page_title="🎓 Student Performance Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# 🔥 S-Class Dark Theme CSS - Maximum Flex
st.markdown("""
<style>
    /* Global Dark Theme Foundation */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
        color: #fafafa;
    }
    
    /* Main Content Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styling - Neon Glow Effect */
    h1 {
        color: #00d4ff !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        margin-bottom: 2rem !important;
        background: linear-gradient(45deg, #00d4ff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: #00d4ff !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e2329 0%, #2d3748 100%);
        border-right: 2px solid #00d4ff;
    }
    
    .sidebar .sidebar-content {
        background: transparent;
    }
    
    /* Navigation Radio Buttons */
    .stRadio > div {
        background: rgba(30, 35, 41, 0.8);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid #00d4ff;
        backdrop-filter: blur(10px);
    }
    
    .stRadio > div > label {
        color: #fafafa !important;
        font-weight: 500;
        padding: 0.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stRadio > div > label:hover {
        background: rgba(0, 212, 255, 0.1);
        transform: translateX(5px);
    }
    
    /* Button Styling - Cyberpunk Aesthetic */
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.6);
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Selectbox & Input Styling */
    .stSelectbox > div > div {
        background: rgba(30, 35, 41, 0.9) !important;
        border: 2px solid #00d4ff !important;
        border-radius: 10px !important;
        color: #fafafa !important;
    }
    
    .stSlider > div > div {
        background: rgba(30, 35, 41, 0.9);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metric Cards - Glassmorphism */
    .metric-card {
        background: rgba(30, 35, 41, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00d4ff;
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        background: rgba(30, 35, 41, 0.9);
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid #00d4ff;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(255, 107, 107, 0.1));
        border-radius: 10px;
        border: 1px solid #00d4ff;
        color: #fafafa !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 35, 41, 0.8);
        border-radius: 0 0 10px 10px;
        border: 1px solid #00d4ff;
        border-top: none;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Custom Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #ff6b6b);
        border-radius: 10px;
    }
    
    /* Markdown Text Enhancement */
    .stMarkdown {
        color: #fafafa !important;
    }
    
    /* Code Block Styling */
    .stCodeBlock {
        background: rgba(30, 35, 41, 0.9) !important;
        border: 1px solid #00d4ff !important;
        border-radius: 10px !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(0, 212, 255, 0.1) !important;
        border: 1px solid #00d4ff !important;
        color: #00d4ff !important;
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.1) !important;
        border: 1px solid #ff6b6b !important;
        color: #ff6b6b !important;
    }
</style>
""", unsafe_allow_html=True)

# 🎯 Enhanced Data Loading with Error Handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('StudentsPerformance.csv')
        return data
    except FileNotFoundError:
        st.error("📁 **StudentsPerformance.csv** not found! Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Load dataset
dataset = load_data()

if dataset is None:
    st.stop()

# 🚀 Advanced Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #00d4ff; text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);'>
            🎯 Navigation Hub
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    sections = [
        "🏠 Dashboard Home", 
        "📊 Data Overview", 
        "🔍 Exploratory Analysis", 
        "📈 Correlation Matrix", 
        "🤖 ML Predictions", 
        "🎯 Key Insights"
    ]
    
    section = st.radio("**Navigate to:**", sections, key="nav_radio")
    
    # Add some interactive metrics in sidebar
    st.markdown("---")
    st.markdown("### 📊 **Quick Stats**")
    if dataset is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Students", f"{len(dataset):,}", delta="Active")
        with col2:
            st.metric("Subjects", "3", delta="Math, Reading, Writing")

# 🏠 Dashboard Home
if section == "🏠 Dashboard Home":
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1>🎓 Student Performance Analytics Dashboard</h1>
        <p style='font-size: 1.2rem; color: #b0b0b0; margin-top: 1rem;'>
            Unleashing the power of data to understand academic excellence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_math = dataset['math score'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #00d4ff; margin: 0;'>📐 Math Average</h3>
            <h2 style='color: #fafafa; margin: 0.5rem 0;'>{avg_math:.1f}</h2>
            <p style='color: #b0b0b0; margin: 0;'>Out of 100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_reading = dataset['reading score'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #ff6b6b; margin: 0;'>📚 Reading Average</h3>
            <h2 style='color: #fafafa; margin: 0.5rem 0;'>{avg_reading:.1f}</h2>
            <p style='color: #b0b0b0; margin: 0;'>Out of 100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_writing = dataset['writing score'].mean()
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #4ecdc4; margin: 0;'>✍️ Writing Average</h3>
            <h2 style='color: #fafafa; margin: 0.5rem 0;'>{avg_writing:.1f}</h2>
            <p style='color: #b0b0b0; margin: 0;'>Out of 100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_avg = (avg_math + avg_reading + avg_writing) / 3
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: #ffd93d; margin: 0;'>🏆 Overall Average</h3>
            <h2 style='color: #fafafa; margin: 0.5rem 0;'>{total_avg:.1f}</h2>
            <p style='color: #b0b0b0; margin: 0;'>Combined Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive feature showcase
    st.markdown("---")
    st.markdown("### 🎮 **Interactive Features**")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **🔥 What Makes This Dashboard Special:**
        - **Real-time Interactive Visualizations** with Plotly
        - **Advanced Statistical Analysis** with correlation matrices
        - **Machine Learning Predictions** with linear regression
        - **Responsive Dark Theme** optimized for professional use
        - **Multi-dimensional Data Exploration** with parallel coordinates
        """)
    
    with feature_col2:
        # Quick visualization preview
        fig = px.scatter(dataset, x='math score', y='reading score', 
                        color='gender', size='writing score',
                        title="📊 Score Relationships Preview",
                        template="plotly_dark")
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
        st.plotly_chart(fig, use_container_width=True)

# 📊 Data Overview Section (Enhanced)
elif section == "📊 Data Overview":
    st.markdown("# 📊 **Data Overview & Statistics**")
    
    # Data preview with enhanced styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 **Dataset Preview**")
        st.dataframe(dataset.head(10), use_container_width=True)
    
    with col2:
        st.markdown("### 📈 **Dataset Info**")
        info_data = {
            "Metric": ["Total Students", "Features", "Numerical Cols", "Categorical Cols", "Missing Values"],
            "Value": [len(dataset), len(dataset.columns), 3, 5, dataset.isnull().sum().sum()]
        }
        st.dataframe(pd.DataFrame(info_data), use_container_width=True)
    
    # Enhanced statistics
    st.markdown("### 📊 **Detailed Statistics**")
    stats_df = dataset.describe().round(2)
    st.dataframe(stats_df, use_container_width=True)

diff > 0 else 'F'} Higher")
        
        with col3:
            writing_diff = dataset.groupby('gender')['writing score'].mean().diff().iloc[-1]
            st.metric("✍️ Writing Score Gap", f"{abs(writing_diff):.1f}", f"{'M' if writing_diff > 0 else 'F'} Higher")
    
    elif analysis_type == "Preparation Impact":
        st.markdown("### 🎯 **Test Preparation Course Impact Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Preparation impact on all subjects
            prep_comparison = dataset.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean()
            
            fig = go.Figure()
            subjects = ['math score', 'reading score', 'writing score']
            colors = ['#00d4ff', '#ff6b6b', '#4ecdc4']
            
            for i, subject in enumerate(subjects):
                fig.add_trace(go.Bar(
                    x=prep_comparison.index,
                    y=prep_comparison[subject],
                    name=subject.replace(' score', '').title(),
                    marker_color=colors[i],
                    opacity=0.8
                ))
            
            fig.update_layout(
                title="🚀 Test Prep Course Impact",
                xaxis_title="Test Preparation Status",
                yaxis_title="Average Score",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot showing score distributions
            fig = px.violin(dataset, x='test preparation course', y='math score',
                          title="📊 Math Score Distribution by Test Prep",
                          template="plotly_dark",
                          color='test preparation course',
                          color_discrete_map={'completed': '#4ecdc4', 'none': '#ff6b6b'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Preparation effectiveness metrics
        st.markdown("### 📈 **Preparation Effectiveness Metrics**")
        col1, col2, col3 = st.columns(3)
        
        prep_impact = dataset.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean()
        
        with col1:
            math_improvement = prep_impact.loc['completed', 'math score'] - prep_impact.loc['none', 'math score']
            st.metric("📐 Math Improvement", f"+{math_improvement:.1f}", "Points")
        
        with col2:
            reading_improvement = prep_impact.loc['completed', 'reading score'] - prep_impact.loc['none', 'reading score']
            st.metric("📚 Reading Improvement", f"+{reading_improvement:.1f}", "Points")
        
        with col3:
            writing_improvement = prep_impact.loc['completed', 'writing score'] - prep_impact.loc['none', 'writing score']
            st.metric("✍️ Writing Improvement", f"+{writing_improvement:.1f}", "Points")
    
    elif analysis_type == "Multi-Factor Analysis":
        st.markdown("### 🎭 **Multi-Factor Performance Analysis**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parallel coordinates plot
            fig = px.parallel_coordinates(
                dataset,
                dimensions=['math score', 'reading score', 'writing score'],
                color='math score',
                color_continuous_scale='viridis',
                title="🌈 Parallel Coordinates: All Scores",
                template="plotly_dark"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Combined factor analysis
            fig = px.scatter(dataset, x='math score', y='reading score',
                           size='writing score', color='gender',
                           symbol='test preparation course',
                           title="🎯 Multi-Dimensional Score Analysis",
                           template="plotly_dark",
                           hover_data=['writing score', 'lunch'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Factor interaction analysis
        st.markdown("### 🔄 **Factor Interactions**")
        
        # Create a comprehensive analysis
        factor_analysis = dataset.groupby(['gender', 'test preparation course', 'lunch']).agg({
            'math score': 'mean',
            'reading score': 'mean',
            'writing score': 'mean'
        }).round(1).reset_index()
        
        # Sunburst chart for hierarchical factor analysis
        fig = px.sunburst(
            factor_analysis,
            path=['gender', 'test preparation course', 'lunch'],
            values='math score',
            title="🌟 Hierarchical Factor Analysis (Math Scores)",
            template="plotly_dark",
            color='math score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top and bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 **Top Performers**")
            dataset['total_score'] = dataset['math score'] + dataset['reading score'] + dataset['writing score']
            top_performers = dataset.nlargest(5, 'total_score')[['gender', 'test preparation course', 'lunch', 'total_score']]
            st.dataframe(top_performers, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 **Improvement Opportunities**")
            bottom_performers = dataset.nsmallest(5, 'total_score')[['gender', 'test preparation course', 'lunch', 'total_score']]
            st.dataframe(bottom_performers, use_container_width=True)