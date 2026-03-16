from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from predict import predict_audio

st.set_page_config(page_title='Audio Event Detection', page_icon='🔊', layout='centered')

st.title('🔊 Audio Event Detection')
st.write('Upload an audio file and the model will try to identify the sound event.')

uploaded_file = st.file_uploader('Choose an audio file', type=['wav', 'mp3', 'ogg', 'flac'])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    try:
        label, confidence = predict_audio(temp_path)
        st.success(f'Predicted event: {label}')
        st.info(f'Confidence: {confidence:.2%}')
    except Exception as exc:
        st.error(str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

st.markdown('---')
st.write('First run `python train.py`, then start the app using `streamlit run app.py`.')
