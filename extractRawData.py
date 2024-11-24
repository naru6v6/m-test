import kagglehub


def extractDataset():
    path = kagglehub.dataset_download("rajsahu2004/lacuna-malaria-detection-dataset")
    print("Path to dataset files:", path)
    return path
