import torch.cuda

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001
L1_LAMBDA = 100
NUM_WORKERS = 4


def main():
    print(DEVICE)
    print(torch.__version__)


if __name__ == "__main__":
    main()
