# Dendrite.ai Screening Task for Data Science Intern position
The problem statement associated with this test can be found in the `Screening Test - DS` document.

## Methodology

1. The `algoparams_from_ui.json.rtf` file contains all the necessary parameters for building our entire Sklearn pipeline therefore it is first converted into a JSON file for easier access.

2. Now, the JSON file is read and the pipeline parameters are determined accordingly.

3. The dataset `iris.csv` is present in this repository itself. It is loaded.

4. The task type is determined from the `algoparams_from_ui.json` file and the target feature is extracted from the dataset.

5. This involves creating a custom transformer that forms the first layer in our pipeline, the **Feature Handling** layer.

6. This involves creating a **Feature Reduction** layer which also uses a custom transformer to perform feature selection/feature 
extraction as required.

7. This involves creating an **Estimator** layer which contains the Estimator (Regressor/Classifier) we are required to use.

8. Fit the Training data into the pipeline and perform GridSearchCV to determine the best hyperparameters for the estimator. We save the best estimator with the best parameters. (based on a performance factor such as  `RMSE` or `Accuracy`)

9. Perform predictions on the Test set and print out the results.

10. Repeat steps 5 to 9 for all (valid) algorithms where `is_selected = True`. 

## Execution Procedure

> Please make sure you have the packages mentioned in the file `requirements.txt` installed, before you execute the `main.py` file. **Use the command `pip install -r requirements.txt`** to do so.

> It is also suggested to disable warnings while executing the program. **Execute the program file `main.py` using the command `python -W "ignore" main.py`**.

