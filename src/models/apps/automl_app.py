import streamlit
 
import pandas as pd
import os

import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report

import plotly.express as pex
from operator import index

from pycaret.regression import (setup, 
                                compare_models, 
                                pull, 
                                save_model,
                                load_model)


if os.path.exists("./dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)


with streamlit.sidebar:
    streamlit.title("test")
    choice = streamlit.radio("Navigation", ["Upload", "Profiling data", "Modelling", "Download"])
    streamlit.info("hi " + "\n<af>"*100 )


if choice == "Upload":

    streamlit.title("j")

    file = streamlit.file_uploader("Upload you data ")

    if file:
        df = pd.read_csv(file, index_col=None)

        df.to_csv("dataset.csv", index=None)
        streamlit.dataframe(df)



if choice == "Profiling data":

    streamlit.title("j")
#
    prof = df.profile_report()
    st_profile_report(prof)

if choice == "Modelling":

    targ = streamlit.selectbox("Choose target column", df.columns)

    if streamlit.button("Run"):

        setup(df, target=targ, system_log=True)
        setup_df = pull()
        streamlit.dataframe(setup_df)

        best = compare_models()
        compare_df = pull()
        streamlit.dataframe(compare_df)

        save_model(best, "best_model")



if choice == "Download":
    with open("best_model.pkl", "rb") as model_file:
        streamlit.download_button("Download best model", model_file, file_name="best_model.pkl")



