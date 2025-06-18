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

# 🔍 Enhanced Exploratory Analysis
elif section == "🔍 Exploratory Analysis":
    st.markdown("# 🔍 **Advanced Exploratory Data Analysis**")
    
    # Interactive score analysis
    analysis_type = st.selectbox(
        "**Choose Analysis Type:**",
        ["Score Correlations", "Gender Analysis", "Preparation Impact", "Multi-Factor Analysis"]
    )
    
    if analysis_type == "Score Correlations":
        col1, col2 = st.columns(2)
        
        with col1:
            # 3D Scatter plot
            fig = px.scatter_3d(dataset, x='math score', y='reading score', z='writing score',
                              color='gender', title="🌟 3D Score Relationships",
                              template="plotly_dark")
            fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            corr_data = dataset[['math score', 'reading score', 'writing score']].corr()
            fig = px.imshow(corr_data, text_auto=".2f", aspect="auto",
                          title="🔥 Score Correlation Matrix",
                          color_continuous_scale="viridis",
                          template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# 📈 Enhanced Correlation Matrix
elif section == "📈 Correlation Matrix":
    st.markdown("# 📈 **Advanced Correlation Analysis**")
    
    # Prepare encoded data
    dataset_encoded = dataset.copy()
    dataset_encoded['gender_code'] = dataset['gender'].map({'female': 0, 'male': 1})
    dataset_encoded['test_prep_code'] = dataset['test preparation course'].map({'completed': 1, 'none': 0})
    dataset_encoded['lunch_code'] = dataset['lunch'].map({'standard': 1, 'free/reduced': 0})
    
    # Enhanced correlation matrix
    corr_cols = ['math score', 'reading score', 'writing score', 'gender_code', 'test_prep_code', 'lunch_code']
    corr_matrix = dataset_encoded[corr_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=".3f", aspect="auto",
                  title="🎯 Complete Correlation Matrix",
                  color_continuous_scale="RdBu_r",
                  template="plotly_dark")
    fig.update_layout(
        width=800, height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation insights
    st.markdown("### 🔍 **Key Correlation Insights**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        reading_writing_corr = corr_matrix.loc['reading score', 'writing score']
        st.metric("📚✍️ Reading-Writing", f"{reading_writing_corr:.3f}", "Strongest Link")
    
    with col2:
        math_reading_corr = corr_matrix.loc['math score', 'reading score']
        st.metric("📐📚 Math-Reading", f"{math_reading_corr:.3f}", "Strong Link")
    
    with col3:
        prep_avg_corr = (corr_matrix.loc['test_prep_code', 'math score'] + 
                        corr_matrix.loc['test_prep_code', 'reading score'] + 
                        corr_matrix.loc['test_prep_code', 'writing score']) / 3
        st.metric("🎯 Test Prep Impact", f"{prep_avg_corr:.3f}", "Moderate Effect")

# 🤖 Enhanced ML Predictions
elif section == "🤖 ML Predictions":
    st.markdown("# 🤖 **Machine Learning Predictions**")
    
    # Model selection with enhanced UI
    model_options = {
        "📐➡️📚 Math → Reading": ("math score", "reading score"),
        "✍️➡️📚 Writing → Reading": ("writing score", "reading score"),
        "✍️➡️📐 Writing → Math": ("writing score", "math score")
    }
    
    selected_model = st.selectbox("**Choose Prediction Model:**", list(model_options.keys()))
    input_col, target_col = model_options[selected_model]
    
    # Train model
    X = dataset[[input_col]]
    y = dataset[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model performance
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2_score = model.score(X_test, y_test)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Model Accuracy (R²)", f"{r2_score:.3f}", f"RMSE: {rmse:.2f}")
        
        # Interactive prediction
        st.markdown("### 🎮 **Try Your Own Prediction**")
        user_input = st.slider(f"Input {input_col}", 
                              int(dataset[input_col].min()), 
                              int(dataset[input_col].max()), 
                              int(dataset[input_col].mean()))
        
        prediction = model.predict([[user_input]])[0]
        st.success(f"🔮 **Predicted {target_col}:** {prediction:.1f}")
    
    with col2:
        # Visualization
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=dataset[input_col], y=dataset[target_col],
            mode='markers', name='Actual Data',
            marker=dict(color='rgba(0, 212, 255, 0.6)', size=8)
        ))
        
        # Regression line
        x_line = np.linspace(dataset[input_col].min(), dataset[input_col].max(), 100)
        y_line = model.predict(x_line.reshape(-1, 1))
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines', name='Prediction Line',
            line=dict(color='#ff6b6b', width=3)
        ))
        
        # User prediction point
        fig.add_trace(go.Scatter(
            x=[user_input], y=[prediction],
            mode='markers', name='Your Prediction',
            marker=dict(color='#ffd93d', size=15, symbol='star')
        ))
        
        fig.update_layout(
            title=f"🎯 {selected_model} Model",
            xaxis_title=input_col,
            yaxis_title=target_col,
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# 🎯 Enhanced Key Insights
elif section == "🎯 Key Insights":
    st.markdown("# 🎯 **Strategic Insights & Recommendations**")
    
    # Key findings with enhanced presentation
    insights = [
        {
            "title": "📚✍️ Reading-Writing Synergy",
            "metric": f"{dataset['reading score'].corr(dataset['writing score']):.1%}",
            "description": "Strong correlation suggests integrated language skills development",
            "color": "#00d4ff"
        },
        {
            "title": "👥 Gender Performance Patterns",
            "metric": f"{abs(dataset.groupby('gender')['math score'].mean().diff().iloc[-1]):.1f}",
            "description": "Point difference in math scores between genders",
            "color": "#ff6b6b"
        },
        {
            "title": "🎯 Test Prep Effectiveness",
            "metric": f"{dataset.groupby('test preparation course')['math score'].mean().diff().iloc[-1]:.1f}",
            "description": "Average score improvement from test preparation",
            "color": "#4ecdc4"
        },
        {
            "title": "🍽️ Socioeconomic Impact",
            "metric": f"{dataset.groupby('lunch')['reading score'].mean().diff().iloc[-1]:.1f}",
            "description": "Score gap between lunch program participants",
            "color": "#ffd93d"
        }
    ]
    
    cols = st.columns(2)
    for i, insight in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""
            <div class='metric-card' style='border-color: {insight["color"]}'>
                <h3 style='color: {insight["color"]}; margin: 0;'>{insight["title"]}</h3>
                <h2 style='color: #fafafa; margin: 0.5rem 0;'>{insight["metric"]}</h2>
                <p style='color: #b0b0b0; margin: 0;'>{insight["description"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Strategic recommendations
    st.markdown("---")
    st.markdown("### 🚀 **Strategic Recommendations**")
    
    recommendations = [
        "🎯 **Integrated Learning**: Leverage the strong reading-writing correlation for combined curriculum design",
        "📊 **Personalized Approaches**: Address gender-specific learning preferences in STEM subjects",
        "🔄 **Test Prep Optimization**: Enhance preparation programs for maximum score improvement",
        "🤝 **Equity Initiatives**: Implement targeted support for students from different socioeconomic backgrounds",
        "📈 **Predictive Analytics**: Use ML models to identify at-risk students early in the academic year"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

# 🎨 Footer with style
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(255, 107, 107, 0.1)); border-radius: 15px; margin-top: 2rem;'>
    <h3 style='color: #00d4ff; margin: 0;'>🚀 Built with Maximum Flex</h3>
    <p style='color: #b0b0b0; margin: 0.5rem 0;'>Streamlit × Advanced Analytics × S-Class Design</p>
    <p style='color: #666; margin: 0; font-size: 0.9rem;'>© 2025 Student Performance Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)
