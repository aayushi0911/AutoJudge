import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack

st.set_page_config(page_title="Problem Difficulty Predictor",layout="wide")

@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl","rb") as f:
        tfidf=pickle.load(f)
    with open("classification_model.pkl","rb") as f:
        clf=pickle.load(f)
    with open("regression_model.pkl","rb") as f:
        reg=pickle.load(f)
    with open("label_encoder.pkl","rb") as f:
        le=pickle.load(f)
    return tfidf,clf,reg,le

tfidf,clf,reg,le=load_models()

st.title("Problem Difficulty Predictor")
st.divider()

problem_desc=st.text_area("Problem Description",height=180)
col1,col2=st.columns(2)

with col1:
    input_desc=st.text_area("Input Description",height=180)

with col2:
    output_desc=st.text_area("Output Description",height=180)

keywords=["array","recursion","binary search","tree","graph","dp","stack","queue","greedy",
          "matrix","set","map","heap","hash","sort"]

if st.button("Predict",type="primary"):
    text=f"{problem_desc}\n{input_desc}\n{output_desc}".strip()

    if not text:
        st.warning("Please enter at least one field.")
    else:
        text_len=len(text)
        sym_cnt=sum(text.count(c) for c in "0123456789+-*/=.$%()^|&")/max(text_len,1)
        word_cnt=len(text.split())
        unique_words=len(set(text.split()))

        numeric_feats=np.array([[text_len,sym_cnt,word_cnt,unique_words]])
        keyword_feats=np.array([[int(k in text.lower()) for k in keywords]])

        X_tf=tfidf.transform([text])

        X_class=hstack([X_tf,numeric_feats,keyword_feats])
        class_pred=clf.predict(X_class)[0]
        class_name=le.inverse_transform([class_pred])[0]

        class_feat=np.array([[class_pred]])
        X_reg=hstack([X_tf,numeric_feats,keyword_feats,class_feat])
        score=reg.predict(X_reg)[0]
        score=max(1.0,min(10.0,score))

        col1,col2=st.columns(2)
        with col1:
            st.metric("Difficulty Class",class_name.title())
        with col2:
            st.metric("Difficulty Score",f"{score:.2f}")

st.divider()
