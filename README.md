# Dendrite.ai Screening Task for Data Science Intern position
The problem statement associated with this test can be found in the `Screening Test - DS` document.

## Methodology of implementation.

**Step [ 1 ]:** The `algoparams_from_ui.json.rtf` file contains all the necessary parameters for building our entire Sklearn pipeline therefore it is first converted into a JSON file for easier access.

**Step [ 2 ]:** Now, the JSON file is read and the pipeline parameters are determined accordingly.

**Step [ 3 ]:** The dataset `iris.csv` is present in this repository itself. It is loaded.

**Step [ 4 ]:** The task type is determined from the `algoparams_from_ui.json` file and the target feature is extracted from the dataset.

**Step [ 5 ]:** This involves creating a custom transformer that forms the first layer in our pipeline, the **Feature Handling** layer.

**Step [ 6 ]:** This involves creating a **Feature Reduction** layer which also uses a custom transformer to perform feature selection/feature 
extraction as required.

**Step [ 7 ]:** This involves creating an **Estimator** layer which contains the Estimator (Regressor/Classifier) we are required to use.

**Step [ 8 ]:** Fit the Training data into the pipeline and perform GridSearchCV to determine the best hyperparameters for the estimator.

**Step [ 9 ]:** Perform predictions on the Test set and print out the results.

**Step [ 10 ]:** Repeat steps 5 to 9 for all (valid) algorithms where `is_selected = True` 

Please make sure you have the libraries mentioned in the file `requirements.txt` installed, before you execute the main.py file.
