from utils import mnist_reader
from utils.download import download
import random
import math
import pickle
import json


def main():
    train_classes_count = 6
    total_classes_count = 10

    for f in range(5):
        # Randomly pick train classes
        all_classes = [x for x in range(total_classes_count)]
        random.shuffle(all_classes)
        train_classes = all_classes[:train_classes_count]
        rest_classes = [x for x in all_classes if x not in train_classes]

        print("Openness table:")
        with open('class_table_fold_%d.txt' % f, 'w') as outfile:
            table = []
            for i in range(total_classes_count - train_classes_count + 1):
                test_target_classes = train_classes + rest_classes[:i]
                openness = 1.0 - math.sqrt(2 * len(train_classes) / (len(train_classes) + len(test_target_classes)))
                print("\tOpenness: %f" % openness)
                table.append({"train": train_classes, "test_target": test_target_classes})
            json.dump(table, outfile, indent=4)

if __name__ == '__main__':
    main()
