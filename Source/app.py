import streamlit as st
from pathlib import Path
import os
from ./DatasetCuration/Curation.py import curate_dataset

def get_file_count(path):
    directory_path = Path(path)
    files = [p for p in directory_path.iterdir() if p.is_file()]
    return len(files)

st.set_page_config(page_title="Data Qualtiy Platform", layout="wide")

def main():
    st.title("Data Quality Platform")
    col1, col2 = st.columns(2)

    with col1:
        st.header("1) Synthesize dataset")
        st.caption("Select the folder containing dataset. They will be split into sub-images for recognition tasks")
        path_to_dataset_raw = st.text_input('Path to dataset', placeholder='Path to dataset')
        dataset_raw_count = get_file_count(path_to_dataset_raw)
        st.write(f'Images detected: {dataset_raw_count}')
        path_to_synthetic_dataset = st.text_input('Path to synthetic dataset', placeholder='Path to synthetic dataset')
        if st.button('Synthesize Dataset'):
            if dataset_raw_count > 0:
                synthesized_dataset_count = 0
                st.success('Dataset Synthesis Complete!')
                st.info(f'Images Generated: {synthesized_dataset_count}')
            else:
                st.error('Please select a valid folder with images')

    with col2:
        st.header("2) Data Labelling")
        st.caption("Create dataset with labels - blur, brightness, contrast, prediction, confidence")
        st.write('Blur - Calculated using Laplacian Variance')
        st.write('Brightness - Value channel in HSV')
        st.write('Contrast - Std. deviation of pixel intensity')
        st.write('Prediction - Prediction of the orginal model')
        st.write('Confidence - Avg. sentence confidence of the prediction')
        if st.button('Start Labelling'):
            if True:
                st.success('Dataset labelling completed!')
                path_to_dataset = ''
                st.info(f'Labelled dataset at: {path_to_dataset}')
            else:
                st.error('Error labelling dataset')

    with col1:
        quality_thresholds = {}
        st.header("3) Dataset Curation")
        st.caption("Curate the dataset to required quality specification")
        with st.container(horizontal=True, vertical_alignment="bottom", gap='small'):
            st.text('Laplacian Variance')
            quality_thresholds['laplacian_var'] = {}
            quality_thresholds['laplacian_var']['threshold_operator'] = st.selectbox('', ['Greater than', 'Lesser than', 'In range'], index=0, key='laplacian')
            quality_thresholds['laplacian_var']['threshold_value'] = st.text_input('', placeholder='Enter threshold here', key='laplacian_text')
        with st.container(horizontal=True, vertical_alignment="bottom", gap='small'):
            st.text('Brightness')
            quality_thresholds['brightness'] = {}
            quality_thresholds['brightness']['threshold_operator'] = st.selectbox('', ['Greater than', 'Lesser than', 'In range'], index=0, key='brightness')
            quality_thresholds['brightness']['threshold_value'] = st.text_input('', placeholder='Enter threshold here', key='brightness_text')
        with st.container(horizontal=True, vertical_alignment="bottom", gap='small'):
            st.text('Contrast')
            quality_thresholds['contrast'] = {}
            quality_thresholds['contrast']['threshold_operator'] = st.selectbox('', ['Greater than', 'Lesser than', 'In range'], index=0, key='contrast')
            quality_thresholds['contrast']['threshold_value'] = st.text_input('', placeholder='Enter threshold here', key='contrast_text')
        with st.container(horizontal=True, vertical_alignment="bottom", gap='small'):
            st.text('Confidence')
            quality_thresholds['confidence'] = {}
            quality_thresholds['confidence']['threshold_operator'] = st.selectbox('', ['Greater than', 'Lesser than', 'In range'], index=0, key='confidence')
            quality_thresholds['confidence']['threshold_value'] = st.text_input('', placeholder='Enter threshold here', key='confidence_text')
        print(quality_thresholds)
if __name__ == "__main__":
    main()