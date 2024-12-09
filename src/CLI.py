import argparse
import src.model as model
import os
import src.utils as utils

path, database_setting = utils.load_config_json()

def main():
    parser = argparse.ArgumentParser(
        description="""auto-classify-images CLI
This program is for auto-classifying images in a folder using the trained model

Follow this simple steps:
    1. Set your path to the project folder or elsewhere. (It will be your storage for images, config, and model)
    2. Put your initially classified images (at least 100 images for each folder in /dataset)
    3. Train the model by running the train command (it will take some time, so be patient)
    4. Put images to be classified to /unclassified or elsewhere
    5. Run the classify command with the folder path to classify the images (default is /unclassified)""", 
        formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")
    
    # classify
    classify_parser = subparsers.add_parser("classify", help="Classify images in a folder using the trained model")
    classify_parser.add_argument("-f", "--folder", type=str, default=os.path.join(path, "storage/unclassified"), required=False, help="Folder path containing images to classify")

    # train
    train_parser = subparsers.add_parser("train", help="Train model. at least 100 images are recommended for each class")
    train_parser.add_argument("-e", "--epochs", type=int, default=5, required=False, help="Number of epochs for training (default epoch: 5)")

    # change path
    path_parser = subparsers.add_parser("path", help="Change path of the storage and setup folders.")
    path_parser.add_argument("-p", "--path", type=str, 
        help="Change path of the storage and setup folders")
    

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        # train function with additional arguments
        model.make_model(num_epochs=args.epochs)

    if args.command == "classify":
        device_model = model.load_model()
        # classify function
        model.classify_from_folder(device_model, args.folder)

    if args.command == "path" and args.path:
        # change path function and setup folders
        utils.create_config_json(args.path)
        utils.setup_storage(args.path)
    
