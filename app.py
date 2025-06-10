import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Set page configuration for a professional look
st.set_page_config(page_title="Student Performance Analysis", layout="wide", page_icon="📚")

# Custom CSS for S-class aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1f2a44;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
dataset = pd.read_csv('StudentsPerformance.csv')

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = ["Home", "Data Overview", "Exploratory Data Analysis", "Correlation Analysis", "Regression Models", "Conclusions"]
section = st.sidebar.radio("Go to:", sections)

# Title and introduction
if section == "Home":
    st.title("📚 Student Performance Analysis Dashboard")
    st.markdown("""
        Welcome to the **Student Performance Analysis Dashboard**! This app explores a dataset of student scores in math, reading, and writing, analyzing factors like gender, test preparation, and ethnicity. Through interactive visualizations and regression models, we uncover insights into academic performance.

        Use the sidebar to navigate through different sections of the analysis.
    """)

# Data Overview
# Replace the "Data Overview" section in app.py
elif section == "Data Overview":
    st.title("Data Overview")
    st.markdown("### Understanding the Dataset")
    st.write("This section provides a preview of the student performance dataset, basic statistical summaries, and detailed information about its structure.")

    # Handle dataset loading with error checking
    try:
        # Dataset Preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Preview")
            st.write("A glimpse of the first few rows of the dataset.")
            st.dataframe(dataset.head(), use_container_width=True)

        # Basic Statistics
        with col2:
            st.subheader("Basic Statistics")
            st.write("Summary statistics for numerical columns (math, reading, and writing scores).")
            stats_df = dataset.describe().round(2)  # Round for readability
            st.dataframe(stats_df, use_container_width=True)

        # Dataset Information
        st.subheader("Dataset Information")
        st.write("Details about columns, data types, and non-null counts.")
        # Create a DataFrame for dataset info
        info_df = pd.DataFrame({
            'Column': dataset.columns,
            'Data Type': dataset.dtypes.values,
            'Non-Null Count': dataset.notnull().sum().values,
            'Missing Values': dataset.isnull().sum().values
        })
        st.dataframe(info_df, use_container_width=True)

        # Additional Info
        st.markdown(f"""
            **Dataset Size**: {dataset.shape[0]} rows, {dataset.shape[1]} columns  
            **Columns**: {', '.join(dataset.columns)}  
            **Numerical Columns**: math score, reading score, writing score  
            **Categorical Columns**: gender, race/ethnicity, parental level of education, lunch, test preparation course
        """)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.write("Please ensure 'StudentsPerformance.csv' is in the same directory as the app and has the expected structure.")
# Exploratory Data Analysis



# ... (other imports and app code remain unchanged)

elif section == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("### Uncovering Insights Through Visualizations")
    st.write("This section dives deep into the student performance dataset, exploring relationships between scores (math, reading, writing) and factors like gender, race/ethnicity, parental education, lunch, and test preparation. Interactive graphs reveal patterns, distributions, and correlations.")

    # Score Relationships
    with st.expander("Score Relationships", expanded=True):
        st.subheader("Pairwise Score Relationships")
        st.write("Explore correlations between math, reading, and writing scores using scatter plots and a pair plot.")
        score_option = st.selectbox("Color Scatter By:", ["None", "Gender", "Test Preparation Course", "Lunch", "Race/Ethnicity", "Parental Level of Education"], key="score_color")
        score_pairs = st.selectbox("Select Score Pair:", ["Math vs. Reading", "Reading vs. Writing", "Math vs. Writing"], key="score_pair")
        
        # Scatter Plot
        x_map = {"Math vs. Reading": "math score", "Reading vs. Writing": "reading score", "Math vs. Writing": "math score"}
        y_map = {"Math vs. Reading": "reading score", "Reading vs. Writing": "writing score", "Math vs. Writing": "writing score"}
        x_col, y_col = x_map[score_pairs], y_map[score_pairs]
        if score_option == "None":
            fig = px.scatter(dataset, x=x_col, y=y_col, title=f"{x_col.capitalize()} vs. {y_col.capitalize()} Scores",
                             trendline="ols", trendline_color_override="#FF4C4C")
        else:
            fig = px.scatter(dataset, x=x_col, y=y_col, color=score_option.lower(), 
                             title=f"{x_col.capitalize()} vs. {y_col.capitalize()} Scores by {score_option}",
                             trendline="ols", trendline_color_override="#FF4C4C")
        fig.update_layout(plot_bgcolor="white", font=dict(color="#1f2a44"))
        st.plotly_chart(fig, use_container_width=True)
        corr = dataset[x_col].corr(dataset[y_col]) * 100
        st.markdown(f"**Insight**: {x_col.capitalize()} and {y_col.capitalize()} have a {corr:.2f}% correlation. {score_option} reveals nuanced patterns (e.g., gender differences in score clusters).")

        # Pair Plot
        st.write("Pair plot of all scores for a holistic view.")
        fig = px.scatter_matrix(dataset, dimensions=["math score", "reading score", "writing score"],
                                color=score_option.lower() if score_option != "None" else None,
                                title="Pair Plot of Scores", height=600)
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(plot_bgcolor="white", font=dict(color="#1f2a44"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight**: Reading and writing scores show the tightest clustering (95% correlation), while math scores are more spread, especially for lower values.")

    # Score Distributions
    with st.expander("Score Distributions"):
        st.subheader("Score Distributions by Category")
        st.write("Box and violin plots show how scores vary across categorical variables.")
        cat_var = st.selectbox("Select Category:", ["Gender", "Race/Ethnicity", "Parental Level of Education", "Lunch", "Test Preparation Course"], key="dist_cat")
        score_type = st.selectbox("Select Score:", ["Math Score", "Reading Score", "Writing Score"], key="dist_score")
        score_col = score_type.lower().replace(" ", " ")
        
        # Box Plot
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(dataset, x=cat_var.lower(), y=score_col, color=cat_var.lower(),
                         title=f"{score_type} Distribution by {cat_var}", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(plot_bgcolor="white", font=dict(color="#1f2a44"))
            st.plotly_chart(fig, use_container_width=True)
        
        # Violin Plot
        with col2:
            fig = px.violin(dataset, x=cat_var.lower(), y=score_col, color=cat_var.lower(), box=True,
                            title=f"{score_type} Distribution by {cat_var}", color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(plot_bgcolor="white", font=dict(color="#1f2a44"))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight**: {score_type} varies across {cat_var.lower()}. For example, females tend to have higher reading/writing scores, while test preparation slightly boosts all scores.")

        # Histogram
        st.write("Histogram of scores, optionally filtered by category.")
        filter_cat = st.multiselect(f"Filter {cat_var}:", dataset[cat_var.lower()].unique(), key="hist_filter")
        filtered_data = dataset if not filter_cat else dataset[dataset[cat_var.lower()].isin(filter_cat)]
        fig = px.histogram(filtered_data, x=score_col, color=cat_var.lower() if filter_cat else None,
                           title=f"{score_type} Histogram", nbins=30, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(plot_bgcolor="white", font=dict(color="#1f2a44"), bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight**: The distribution of {score_type.lower()} shows {cat_var.lower()} differences, with some groups (e.g., test prep completers) skewing higher.")

    # Categorical Comparisons
    with st.expander("Categorical Comparisons"):
            st.subheader("Mean Scores by Categorical Variables")
            st.write("Bar plots compare average scores across categories, with error bars showing standard deviation for variability.")
            cat_var = st.selectbox("Select Category:", ["Gender", "Race/Ethnicity", "Parental Level of Education", "Lunch", "Test Preparation Course"], key="bar_cat")
            
            # Bar Plot with Error Bars
            try:
                mean_scores = dataset.groupby(cat_var.lower())[["math score", "reading score", "writing score"]].agg(["mean", "std"]).reset_index()
                fig = make_subplots(rows=1, cols=3, subplot_titles=("Math Score", "Reading Score", "Writing Score"))
                for i, score in enumerate(["math score", "reading score", "writing score"], 1):
                    fig.add_trace(go.Bar(
                        x=mean_scores[cat_var.lower()], y=mean_scores[(score, "mean")],
                        error_y=dict(type="data", array=mean_scores[(score, "std")], visible=True),
                        name=score, marker_color=["#FF4C4C", "#FF9999", "#FFB6B6"][i-1]
                    ), row=1, col=i)
                fig.update_layout(title=f"Mean Scores by {cat_var}", showlegend=False, plot_bgcolor="white", font=dict(color="#1f2a44"))
                st.plotly_chart(fig, use_container_width=True)
                
                # Tailored Insight
                def get_insight(cat_var, mean_scores):
                    if cat_var.lower() == "gender":
                        math_diff = mean_scores[mean_scores["gender"] == "male"][("math score", "mean")].iloc[0] - mean_scores[mean_scores["gender"] == "female"][("math score", "mean")].iloc[0]
                        read_diff = mean_scores[mean_scores["gender"] == "female"][("reading score", "mean")].iloc[0] - mean_scores[mean_scores["gender"] == "male"][("reading score", "mean")].iloc[0]
                        return f"Females outperform males in reading (+{read_diff:.1f} points) and writing, while males have a slight edge in math (+{math_diff:.1f} points)."
                    elif cat_var.lower() == "race/ethnicity":
                        top_group = mean_scores[("math score", "mean")].idxmax()
                        top_score = mean_scores.loc[top_group, ("math score", "mean")]
                        return f"Group {mean_scores.loc[top_group, 'race/ethnicity']} has the highest average scores, particularly in math ({top_score:.1f}), suggesting potential cultural or socioeconomic factors."
                    elif cat_var.lower() == "parental level of education":
                        top_edu = mean_scores[("reading score", "mean")].idxmax()
                        top_score = mean_scores.loc[top_edu, ("reading score", "mean")]
                        return f"Higher parental education (e.g., {mean_scores.loc[top_edu, 'parental level of education']}) correlates with better scores, especially in reading ({top_score:.1f}), but the effect plateaus at higher levels."
                    elif cat_var.lower() == "lunch":
                        std_score = mean_scores[mean_scores["lunch"] == "standard"][("math score", "mean")].iloc[0]
                        free_score = mean_scores[mean_scores["lunch"] == "free/reduced"][("math score", "mean")].iloc[0]
                        return f"Students with standard lunch score higher across all subjects (e.g., math: {std_score:.1f} vs. {free_score:.1f}), likely reflecting socioeconomic advantages."
                    elif cat_var.lower() == "test preparation course":
                        completed_score = mean_scores[mean_scores["test preparation course"] == "completed"][("math score", "mean")].iloc[0]
                        none_score = mean_scores[mean_scores["test preparation course"] == "none"][("math score", "mean")].iloc[0]
                        return f"Test preparation courses provide a modest boost (e.g., math: {completed_score:.1f} vs. {none_score:.1f}), but the impact is smaller than expected."
                    return "No specific insight available for this category."

                insight = get_insight(cat_var, mean_scores)
                st.markdown(f"**Insight**: {insight}")
            except Exception as e:
                st.error(f"Error rendering bar plot: {e}")

    # Correlation Analysis
    with st.expander("Correlation Analysis"):
        st.subheader("Correlation Heatmap")
        st.write("Visualize correlations between numerical scores and encoded categorical variables.")
        # Encode categorical variables
        dataset_encoded = dataset.copy()
        dataset_encoded["gender_code"] = dataset["gender"].map({"female": 0, "male": 1})
        dataset_encoded["test_prep_code"] = dataset["test preparation course"].map({"completed": 1, "none": 0})
        dataset_encoded["lunch_code"] = dataset["lunch"].map({"standard": 1, "free/reduced": 0})
        corr_matrix = dataset_encoded[["math score", "reading score", "writing score", "gender_code", "test_prep_code", "lunch_code"]].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu", aspect="auto",
                        title="Correlation Heatmap", labels=dict(color="Correlation"))
        fig.update_layout(plot_bgcolor="white", font=dict(color="#1f2a44"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight**: Reading and writing scores have a 95% correlation, while categorical factors (gender, test prep, lunch) show weaker correlations (~20-30%).")

    # Multi-Dimensional Analysis
    with st.expander("Multi-Dimensional Analysis"):
        st.subheader("Parallel Coordinates Plot")
        st.write("Explore relationships across multiple variables simultaneously.")
        cat_var = st.selectbox("Color By:", ["Gender", "Test Preparation Course", "Lunch", "Race/Ethnicity"], key="parallel_cat")
        fig = px.parallel_coordinates(
            dataset_encoded,
            dimensions=["math score", "reading score", "writing score", "gender_code", "test_prep_code", "lunch_code"],
            color=dataset[cat_var.lower()].astype("category").cat.codes,
            labels={"math score": "Math", "reading score": "Reading", "writing score": "Writing",
                    "gender_code": "Gender (0=F, 1=M)", "test_prep_code": "Test Prep (0=None, 1=Completed)",
                    "lunch_code": "Lunch (0=Free, 1=Standard)"},
            title=f"Parallel Coordinates by {cat_var}"
        )
        fig.update_layout(font=dict(color="#1f2a44"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight**: This plot reveals multi-variable interactions. For example, high scores often align with standard lunch and test preparation.")

    # Interaction Effects
    with st.expander("Interaction Effects"):
        st.subheader("Scores by Combined Categories")
        st.write("Examine how two categorical variables interact to affect scores.")
        cat1 = st.selectbox("First Category:", ["Gender", "Test Preparation Course", "Lunch"], key="int_cat1")
        cat2 = st.selectbox("Second Category:", ["Race/Ethnicity", "Parental Level of Education"], key="int_cat2")
        score_type = st.selectbox("Select Score:", ["Math Score", "Reading Score", "Writing Score"], key="int_score")
        score_col = score_type.lower().replace(" ", " ")
        
        # Grouped Bar Plot
        grouped_data = dataset.groupby([cat1.lower(), cat2.lower()])[score_col].mean().reset_index()
        fig = px.bar(grouped_data, x=cat1.lower(), y=score_col, color=cat2.lower(), barmode="group",
                     title=f"Mean {score_type} by {cat1} and {cat2}", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(plot_bgcolor="white", font=dict(color="#1f2a44"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Insight**: Interactions between {cat1.lower()} and {cat2.lower()} reveal nuanced effects. For example, test preparation benefits vary by parental education level.")

    # EDA Summary
    with st.expander("EDA Summary"):
        st.subheader("Key Findings")
        st.markdown("""
            - **Score Correlations**: Reading and writing scores are highly correlated (95%), followed by math-reading (81%) and math-writing (~80%).
            - **Gender Effects**: Females excel in reading/writing, males slightly in math, but differences are modest.
            - **Test Preparation**: Small positive impact on all scores (~20-25% correlation), less effective than expected.
            - **Lunch**: Standard lunch correlates with higher scores, likely reflecting socioeconomic factors.
            - **Race/Ethnicity**: Group differences exist, with some groups (e.g., Group E) consistently scoring higher.
            - **Parental Education**: Higher education levels associate with better scores, but the effect plateaus.
            - **Interactions**: Combining factors (e.g., gender and parental education) reveals complex patterns, such as stronger test prep effects for certain groups.
            - **Distributions**: Scores are roughly normal, with slight right skew for lower performers, varying by category.
        """)
# Correlation Analysis
elif section == "Correlation Analysis":
    st.title("Correlation Analysis")
    st.markdown("### Quantifying Relationships")
    st.write("We calculate correlations to understand the strength of relationships between variables.")

    # Prepare data for correlations
    dataset['gender code'] = dataset['gender'].map({'female': 0, 'male': 1})
    dataset['test preparation course code'] = dataset['test preparation course'].map({'completed': 1, 'none': 0})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Between Scores")
        st.write(f"Math vs. Reading: **{dataset['math score'].corr(dataset['reading score'])*100:.2f}%**")
        st.write(f"Reading vs. Writing: **{dataset['reading score'].corr(dataset['writing score'])*100:.2f}%**")
        st.write(f"Math vs. Writing: **{dataset['math score'].corr(dataset['writing score'])*100:.2f}%**")
        st.markdown("**Insight**: Reading and writing scores have the highest correlation (95%), indicating a strong relationship.")
    
    with col2:
        st.subheader("Impact of Gender and Test Preparation")
        st.write(f"Gender vs. Reading: **{dataset['reading score'].corr(dataset['gender code'])*100:.2f}%**")
        st.write(f"Gender vs. Math: **{dataset['math score'].corr(dataset['gender code'])*100:.2f}%**")
        st.write(f"Gender vs. Writing: **{dataset['writing score'].corr(dataset['gender code'])*100:.2f}%**")
        st.write(f"Test Prep vs. Reading: **{dataset['reading score'].corr(dataset['test preparation course code'])*100:.2f}%**")
        st.write(f"Test Prep vs. Math: **{dataset['math score'].corr(dataset['test preparation course code'])*100:.2f}%**")
        st.write(f"Test Prep vs. Writing: **{dataset['writing score'].corr(dataset['test preparation course code'])*100:.2f}%**")
        st.markdown("**Conclusion**: Gender and test preparation have low correlations with scores, suggesting limited impact.")

# Regression Models
elif section == "Regression Models":
    st.title("Regression Models")
    st.markdown("### Predicting Scores with Linear Regression")
    st.write("We train linear regression models to predict scores based on other scores and evaluate their performance.")

    # Model selection
    model_choice = st.selectbox("Select Model:", ["Math → Reading", "Writing → Reading", "Writing → Math"])

    def train_and_plot_model(input_col, target_col, title):
        input_train, input_test, target_train, target_test = train_test_split(
            dataset[[input_col]], dataset[target_col], test_size=0.1, random_state=42)
        model = LinearRegression()
        model.fit(input_train, target_train)
        predicted = model.predict(input_test)
        
        # Calculate RMSE
        rmse_val = np.sqrt(np.mean((target_test - predicted) ** 2))
        st.write(f"**RMSE**: {rmse_val:.2f} (lower is better)")

        # Plot regression line
        scatter = go.Scatter(x=dataset[input_col], y=dataset[target_col], mode='markers', name='Actual')
        x = list(range(0, 100))
        y = [model.coef_[0] * i + model.intercept_ for i in x]
        line = go.Scatter(x=x, y=y, mode='lines', name='Model Prediction', line=dict(color='red'))
        fig = go.Figure(data=[scatter, line])
        fig.update_layout(title=title, xaxis_title=input_col, yaxis_title=target_col)
        st.plotly_chart(fig, use_container_width=True)

        # Interactive prediction
        st.subheader("Try Your Own Prediction")
        score = st.slider(f"Input {input_col}", 0, 100, 50)
        prediction = model.predict([[score]])[0]
        st.write(f"Predicted {target_col} for {input_col} of {score}: **{prediction:.2f}**")

        return model

    if model_choice == "Math → Reading":
        model = train_and_plot_model('math score', 'reading score', 'Math vs. Reading Score with Regression Line')
        st.markdown(f"**Model Details**: Slope = {model.coef_[0]:.2f} Ministries of Education, Intercept = {model.intercept_:.2f}")
        st.markdown("**Insight**: The model predicts reading scores from math scores with an RMSE of ~8, indicating reasonable accuracy but struggles with outliers.")
    elif model_choice == "Writing → Reading":
        model = train_and_plot_model('writing score', 'reading score', 'Writing vs. Reading Score with Regression Line')
        st.markdown(f"**Model Details**: Slope = {model.coef_[0]:.2f}, Intercept = {model.intercept_:.2f}")
        st.markdown("**Insight**: With an RMSE of ~4.5, this model is highly accurate due to the strong correlation (95%) between writing and reading scores.")
    elif model_choice == "Writing → Math":
        model = train_and_plot_model('writing score', 'math score', 'Writing vs. Math Score with Regression Line')
        st.markdown(f"**Model Details**: Slope = {model.coef_[0]:.2f}, Intercept = {model.intercept_:.2f}")
        st.markdown("**Insight**: The model has a higher RMSE, indicating less predictive accuracy compared to the writing-reading model.")

# Conclusions
elif section == "Conclusions":
    st.title("Key Conclusions")
    st.markdown("""
        ### Summary of Findings
        After exploring the student performance dataset, here are the key insights:

        1. **Score Correlations**:
           - Reading and writing scores have the highest correlation (95%), indicating students proficient in one are likely proficient in the other.
           - Math and reading scores show a strong correlation (81%), suggesting a general academic aptitude across subjects.
           - Math and writing scores are also correlated but to a lesser extent.

        2. **Gender Impact**:
           - Girls tend to outperform boys in reading scores, while boys have a slight edge in math scores.
           - However, gender has a low correlation with overall scores, suggesting it’s not a major determinant of performance.

        3. **Test Preparation**:
           - Test preparation courses show a small positive correlation with scores (~20-25%), but the impact is minimal, suggesting limited effectiveness.

        4. **Regression Models**:
           - The writing-to-reading score model is the most accurate (RMSE ~4.5), reflecting their high correlation.
           - The math-to-reading model is reasonably accurate (RMSE ~8), but struggles with outliers.
           - The writing-to-math model has higher error, indicating more variability.

        ### Final Thoughts
        This analysis highlights the interconnectedness of academic skills, with reading and writing being particularly closely linked. Factors like gender and test preparation have limited impact, suggesting that student performance is driven by other underlying factors not captured in this dataset. The regression models provide a useful predictive tool, especially for reading scores based on writing performance.
    """)

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; color: #666;'>
        Built with Streamlit | Data Source: StudentsPerformance.csv | © 2025 Student Performance Analysis
    </div>
""", unsafe_allow_html=True)