import argparse

def main():
    parser = argparse.ArgumentParser(description="auto-classify-image CLI")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # train
    parser.add_argument("-t", "--train", action="store_true", help="Train model based on ResNet50")

    # classify
    parser.add_argument("-c", "--classify", action="store_true", help="Classify images in a 'classified' folder using the trained model")

    # do setup
    parser.add_argument("-s", "--setup", action="store_true", help="setup folders in main path")


    args = parser.parse_args()

    if args.train:
        # train function

    if args.classify:
        # classify function

    if args.setup:
        # setup function

if __name__ == "__main__":
    main()


