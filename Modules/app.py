import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from preprocessing import preprocess_pipeline
from data_analyzer import pipeline
from model_trainer import run_pipeline

st.title("ML Pipeline Visualizer")
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
st.subheader("Need a try, upload your csv below")
data_path = st.file_uploader("Upload your csv here", type="csv")

if data_path:
    df, data = pipeline(data_path)
    

else :
    st.warning("Please upload a file")

st.divider()

if data_path:
    st.title("Analysis")

    st.header("Sample Data:")
    st.write(df.sample(5))

    st.header("Basic Overview:")
    st.divider()
    cols = st.columns(len(data["basic"]))
    for i, (label, value) in enumerate(data["basic"].items()):
        with cols[i]:
            st.metric(label=label, value=value)

    st.divider()
    st.header("Quality:")
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
    st.header("Columns-wise Analysis:")
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
    st.header("In Depth Report:")
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
    st.header("Suggested Preprocessing Steps:")
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
    
    if middle.button("Run Preprocessing Steps", type = "primary", width = "stretch"):
        df_clean = preprocess_pipeline(df, data)

        st.header("Preprocessed Data:")
        st.write(df_clean.sample(5))
    
        st.divider(width = 10)

        columns = list(df_clean.columns)
        target =st.selectbox(
            "Select your target column next for cross validation:", 
            columns,index=None,  placeholder = "choose an option"
            )
        st.subheader(f"selected target : {target}")




