import json

import streamlit as st
from src.feature_importance import get_feature_importance

st.set_page_config(
    page_title='Топ-5 фичей'
)

st.title('Топ-5 фичей')

top_5_json = get_feature_importance()
j = st.json(top_5_json)

st.download_button('Скачать', json.dumps(top_5_json, ensure_ascii=False), 'top-5-feature-importance.json')
