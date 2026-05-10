**Situation:** Generating high-quality OCR datasets for finetuning resource-constrained edge AI models is traditionally manual & labor-intensive. 

**Task:** I set out to build an automated AI workflow that could curate high-value samples from a raw dataset, serving this as a platform to streamline the transition from raw data to a finetuned model. 

**Action:** The workflow is triggered when a user uploads a generic image dataset. A text detection model first scans the images to generate an initial OCR dataset. Then an OCR model calculates confidence scores. During iteration, I integrated additional logic to capture signals like contrast, Laplacian variance etc. thereby, shifting from simple extraction to a prioritization logic that surfaced only high-value samples, ensuring labelling effort was focused on where it mattered most.

**Result:** This system enabled consistent, repeatable evaluation across varied capture conditions & significantly reduced manual labelling effort. The project was recognized with Second Place at the Baidu AI Developer Challenge & was spotlighted by D-Robotics.


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
