import datetime
import tempfile

import streamlit as st

import src.preprocessing as preprocessing
import src.scorer as scorer

csv = None
st.set_page_config(
    page_title='Предсказание оттока'
)

st.title('Предсказание оттока')
st.header('Загрузите датасет')
uploaded_file = st.file_uploader(' ', 'csv', label_visibility='hidden')

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix='.csv') as f:
        if st.button('Предсказать'):
            with st.spinner('Обработка файла'):
                data = uploaded_file.getvalue()
                f.write(data)
                filename = f.name

            with st.spinner('Препроцессинг'):
                input_df = preprocessing.import_data(filename)
                preprocessed_df = preprocessing.run_preproc(input_df)

            with st.spinner('Предсказание'):
                submission = scorer.make_pred(preprocessed_df, filename)
                csv = scorer.to_csv(submission)

            with st.spinner('Распределение скоров'):
                fig = scorer.kde(submission)
                st.pyplot(fig)

            st.download_button(
                label='Скачать предсказания',
                data=csv,
                file_name='predict_' + datetime.datetime.now().isoformat() + '.csv',
                mime='text/csv',
            )
