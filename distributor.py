import csv
import os
import shutil

source_folder = 'archive/train/images'

destination_folders = ['new_images/cargo', 'new_images/military', 'new_images/carrier', 'new_images/cruise', 'new_images/tanker']

csv_file = 'archive/train/train.csv'

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        image_name = row[0]
        category = row[1]

        if category == '1':
            destination_folder = destination_folders[0]
        elif category == '2':
            destination_folder = destination_folders[1]
        elif category == '3':
            destination_folder = destination_folders[2]
        elif category == '4':
            destination_folder = destination_folders[3]
        elif category == '5':
            destination_folder = destination_folders[4]
        else:
            continue

        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copyfile(source_path, destination_path)
