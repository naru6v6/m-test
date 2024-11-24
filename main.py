import kagglehub
import extractRawData
import transformRawData


if __name__ == "__main__":
    path = extractRawData.extractDataset()
    transformRawData.addPath(path)
