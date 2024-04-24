import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st
from src.cleanData.preProcessing import preProcessing
from src.cleanData.normalize import normalize
from src.decompositionData.decomposition import decomposition
from src.dataViz.vizualisation import vizualisation

st.title("Analyse de Données")

pr = preProcessing()
n = normalize()
d = decomposition()
v = vizualisation()
pr.load()
st.session_state['titre']="Ce Formulaire Prédit le diabète"
st.session_state['df']=pr.df
st.session_state['values']={}
st.session_state['button']= "Envoie"
st.session_state["Query Response"] = "Query Response"
st.sidebar.title(st.session_state['titre'])

st.write(pr.headDF())
st.write(pr.shapeDF())
st.write(pr.infoDF())
st.write(pr.describeDF())
#col1.hist(pr.getDataWithoutLabels())rrr
fig, ax = plt.subplots(nrows=4,ncols=2,figsize = (50,50))
for i,ax in enumerate(ax.flatten()):
    ax.set_title(pr.df.columns[i])
    ax.hist(pr.df[pr.df.columns[i]],20)
st.pyplot(fig)

st.bar_chart(pr.getLabelSeries())
st.write(pr.corrDF())
st.scatter_chart(pr.df,x='Age',y='Pregnancies')
pca = d.Pca(pr.getDataWithoutLabels())
st.scatter_chart(pd.DataFrame(pca,columns=["x","y"]))



for i in st.session_state['df'].columns:
    st.session_state['values'][i] = st.sidebar.selectbox(i, st.session_state['df'][i])
if st.sidebar.button(st.session_state['button']):
    data = requests.get("http://localhost:5000/modelApi", params=st.session_state['values']).json()
    st.session_state['output_query']= data
    st.sidebar.header(st.session_state["Query Response"])
    st.sidebar.write( st.session_state['output_query'])


