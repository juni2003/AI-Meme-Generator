import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.gui import MemeGeneratorApp

def parse_arguments():
    parser = argparse.ArgumentParser(description='Enhanced AI Meme Generator')
    parser.add_argument('--dataset', default='data/labels.csv', help='Path to dataset CSV file')
    parser.add_argument('--images', default='images', help='Path to images folder')
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        import tkinter as tk
        root = tk.Tk()
        app = MemeGeneratorApp(root, args.dataset, args.images)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()