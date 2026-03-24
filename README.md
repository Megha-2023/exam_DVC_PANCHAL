# Exam DVC and Dagshub
Project "Flotation Process": Mineral processing, concentrate silica from ore. 


The project contains:
- Script to split raw data.
- Script to normalize data.
- Script to search best parameter using GridSearch.
- Script to train model based on best parameter.
- Script to evaluate the model and generating scores.
- DVC Pipeline in dvc.yaml file with each stage executing respective script.

To test run following command on the terminal:

```bash
dvc repro
```



