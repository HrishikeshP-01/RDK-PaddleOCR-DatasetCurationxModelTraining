## Instructions to run the progject
1. Clone the repo to the RDK environment
2. Navigate to Source/model & run the download.sh file & make sure both the detection & recognition model are in the Source/model folder
3. Run the app via streamlit: python -m streamlit run app.py
4. Paste all your input images in Data/Input
5. In the web interface enter the input path: Data/Input
6. Enter output path as: Data/Output
7. Click on Synthesize dataset button
8. Once synthesis succeeded make sure the images have been saved to Data/Output
9. Click on the label dataset button
10. Once this is complete choose the values for the parameters in the curation section & click on curate button
11. The datasets will be saved to the parent directory
