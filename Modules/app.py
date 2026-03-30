import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from preprocessing import preprocess_pipeline
from data_analyzer import pipeline
from model_trainer import trainer_pipeline

left, middle, right = st.columns(3)
middle.title(f"Ml_Pipeline_Visualizer", text_alignment = "justify", width = "content")
st.subheader("Developed by YashRawat")
st.markdown(
    """
    This project is developed by a tech enthusiast who wanted to ease the process of data processing and 
    model evaluation. 

    I am still learning Machine Learning and wanted to create a project on my own, so I built this webpage
    where you can upload your data and understand a few common insights that usually takes hours to understand
    hope you like it

    """)
st.divider()
left, middle, right = st.columns(3)
middle.title("Upload Data", text_alignment = "center", width = "content")
st.subheader("Need a try, upload your csv below")
data_path = st.file_uploader("Upload your csv here", type="csv")

if data_path:
    df, data = pipeline(data_path)
    

else :
    st.warning("Please upload a file")

st.divider()

if data_path:
    left, middle, right = st.columns(3)
    middle.title("Analysis")

    middle.header("Sample Data:")
    st.write(df.sample(5))
    left, middle, right = st.columns(3)
    middle.header("Basic Overview:")
    st.divider()
    cols = st.columns(len(data["basic"]))
    for i, (label, value) in enumerate(data["basic"].items()):
        with cols[i]:
            st.metric(label=label, value=value)

    st.divider()
    left, middle, right = st.columns(3)
    middle.header("Quality:")
    st.divider()
    #Total null count
    n_values = data["quality"]["total_null"]
    st.subheader(f"Total Null Values - \n{n_values}")
    #Total duplicate count
    d_values = data["quality"]["duplicate_count"]
    if d_values != 0:
        st.subheader(f"Total Duplicate Values - \n{d_values}")

    col1, col2 = st.columns(2)
    with col1:
        #column-wise null
        st.subheader(f"Column-Wise Null Counts-")
        n_cols = {}
        col_name = ["Column Name", "Null Value"]
        for (key, value) in data["quality"]["n_cols_null"].items():
            if value !=0:
                n_cols[key] = value
        null_df = pd.DataFrame(n_cols.items(), columns = col_name)
        st.dataframe(null_df, hide_index=True)

    with col2:
        #column-wise null percent
        st.subheader(f"Column-Wise Null Percentages-")
        null_percent = pd.DataFrame(data["quality"]["null_percent"].items(), columns= ["Column Name", "Percentage of Null"])
        st.dataframe(null_percent, hide_index = True)


    st.divider()
    left, middle, right = st.columns(3)
    middle.header("Columns-wise Analysis:")
    st.divider()
    col1, col2,col3= st.columns(3)
    with col1:
        st.subheader("Numerical Columns-")
        m_list = ""
        for item in data["columns"]["numeric"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)
    with col2:
        st.subheader("Categorical Columns-")
        m_list = ""
        for item in data["columns"]["categorical"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)
    with col3:
        st.subheader("Unique Value Columns-")
        unique_df = pd.DataFrame(data["columns"]["unique_counts"].items(), columns = ["Column Name", "Unique Values"])
        st.dataframe(unique_df, hide_index= True)


    st.divider()
    left, middle, right = st.columns(3)
    middle.header("In Depth Report:")
    st.divider()
    col1, col2,col3,col4 = st.columns(4)

    with col1:
        st.subheader("Identifier Columns-")
        m_list = ""
        for item in data["analysis"]["identifier_cols"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)
    with col2:
        st.subheader("Constant Columns-")
        m_list = ""
        for item in data["analysis"]["constant_cols"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)
    with col3:
        st.subheader("Low Cardinality-")
        m_list = ""
        for item in data["analysis"]["low_cardinality_numeric"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)
    with col4:
        st.subheader("High Cardinality-")
        m_list = ""
        for item in data["analysis"]["high_cardinality_categorical"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)


    st.divider()
    left, middle, right = st.columns(3)
    middle.header("Suggested Preprocessing Steps:")
    st.divider()
    col1,col2,col3, col4, col5= st.columns(5)
    with col1:
        st.subheader("Drop Columns-")
        m_list = ""
        for item in data["analysis"]["preprocessing"]["drop_cols"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)

    with col2:
        st.subheader("Fill with Mean-")
        m_list = ""
        for item in data["analysis"]["preprocessing"]["fill_mean"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)

    with col3:
        st.subheader("Fill with Mode-")
        m_list = ""
        for item in data["analysis"]["preprocessing"]["fill_mode"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)

    with col4:
        st.subheader("Encode-")
        m_list = ""
        for item in data["analysis"]["preprocessing"]["encode"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)

    with col5:
        st.subheader("Scale- ")
        m_list = ""
        for item in data["analysis"]["preprocessing"]["scale"]:
            m_list += f"* {item}\n"
        st.markdown(m_list)
    
    
    st.divider(width = "stretch")
    left, middle, right = st.columns(3)
    middle.title("Preprocessing")    
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None
    if middle.button("Run Preprocessing Steps", type = "primary", width = "stretch"):
        st.session_state.df_clean = preprocess_pipeline(df, data)
    if st.session_state.df_clean is not None:
        st.header("Preprocessed Data:")
        st.write(st.session_state.df_clean.sample(5))
    
    st.divider()
    if st.session_state.df_clean is not None:
        columns = list(st.session_state.df_clean.columns)
        st.session_state.target =st.selectbox(
            "Select your target column next for cross validation:", 
            columns,index=None,  placeholder = "choose an option"
            )
        if st.session_state.target is not None: 
            st.subheader(f"selected target : {st.session_state.target}")
            left, middle, right = st.columns(3)
            middle.header("Model Evaluation")
            if middle.button("Start Model Evaluation", type = "primary", width = "stretch"):
                st.session_state.problem_type, st.session_state.model_suggested, st.session_state.scoring_metric, st.session_state.result, st.session_state.best_model, st.session_state.score= trainer_pipeline(st.session_state.df_clean, st.session_state.target)

            if st.session_state.problem_type is not None:
                st.subheader(f"Problem Type: {st.session_state.problem_type}", divider=True)                

            if st.session_state.model_suggested is not None:
                st.subheader("Model suggested:")
                mod_list = []
                m_list= ""
                for name, mod in st.session_state.model_suggested:
                    m_list += f"* {name}\n"
                st.markdown(m_list)
            
            if st.session_state.scoring_metric is not None:
                st.subheader(f"Scoring Metric: {st.session_state.scoring_metric}")
            
            if st.session_state.result is not None:
                left,mid,right = st.columns(3)
                mid.subheader("Cross Val Score from suggested models")
                result_df = pd.DataFrame(st.session_state.result.items(), columns = ["Model", "Cross Validation Score"])
                mid.dataframe(result_df, hide_index=True)
         
            
            if st.session_state.best_model is not None:
                st.subheader(f"Best Model: {st.session_state.best_model}")

            if st.session_state.score is not None:
                st.subheader(f"Score: {st.session_state.score}")
            




