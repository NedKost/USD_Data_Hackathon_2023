from src.train import train_model
from src.predict import batch_predict


def main():
    train_model()
    batch_predict()

if __name__ == "__main__":
    main()
    