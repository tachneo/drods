import glob
import xml.etree.ElementTree as ET
from functools import reduce
import pandas as pd
import os
from shutil import copy

# Load all XML files and store in list
xmlfiles = glob.glob('1_dataprepration/data_images/*.xml')
replace_text = lambda x: x.replace('\\', '/')
xmlfiles = [replace_text(x) for x in xmlfiles]

def extract_text(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    # Extract filename
    image_name = root.find('filename').text
    
    # Width and height of the image
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    
    objs = root.findall('object')
    parser = []
    
    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        
        center_x = (xmin + xmax) / 2 / width
        center_y = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax, center_x, center_y, w, h])
    
    return parser

# Apply the extract_text function to each XML file
parser_all = list(map(extract_text, xmlfiles))

# Flatten the list of lists using reduce
data = reduce(lambda x, y: x + y, parser_all)

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax', 'center_x', 'center_y', 'w', 'h'])

# Assign unique ID to each object name
name_to_id = {name: idx for idx, name in enumerate(df['name'].unique(), start=1)}
df['id'] = df['name'].map(name_to_id)

# Display the first few rows of the DataFrame
print(df.head())

# Get unique filenames
images = df['filename'].unique()
print(f"Total number of unique images: {len(images)}")

# Create a DataFrame for the unique filenames
img_df = pd.DataFrame(images, columns=['filename'])

# Shuffle and split the DataFrame into 80% train and 20% test
if len(images) > 1:
    img_train = tuple(img_df.sample(frac=0.8, random_state=42)['filename'])
    img_test = tuple(img_df.query('filename not in @img_train')['filename'])
else:
    img_train = images
    img_test = []

# Ensure there is at least one image in the test set
if len(img_test) == 0 and len(images) > 1:
    img_train = img_train[:-1]
    img_test = (images[-1],)

# Check the number of images in train and test sets
print(f"Number of images in train set: {len(img_train)}")
print(f"Number of images in test set: {len(img_test)}")

# Create train and test DataFrames by filtering the original DataFrame
train_df = df.query('filename in @img_train')
test_df = df.query('filename in @img_test')

# Display the first few rows of the train and test DataFrames
print("Train DataFrame:")
print(train_df.head())

print("Test DataFrame:")
print(test_df.head())

# Create directories for train and test data
train_folder = '1_dataprepration/data_images/train'
test_folder = '1_dataprepration/data_images/test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Group by filename
cols = ['filename', 'id', 'center_x', 'center_y', 'w', 'h']
groupby_obj_train = train_df[cols].groupby('filename')
groupby_obj_test = test_df[cols].groupby('filename')

# Function to copy images and save labels
def save_data(filename, folder_path, group_obj):
    # Copy image
    src = os.path.join('1_dataprepration/data_images', filename)
    dst = os.path.join(folder_path, filename)
    print(f"Copying image from {src} to {dst}")
    if os.path.exists(src):
        copy(src, dst)
        print(f"Image {filename} copied successfully.")
    else:
        print(f"Image {filename} not found at {src}.")
    
    # Save the labels
    text_filename = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
    group_obj.get_group(filename).set_index('filename').to_csv(text_filename, sep=' ', index=False, header=False)
    print(f"Labels for {filename} saved to {text_filename}.")

# Create Series of filenames for train and test sets
filename_series_train = pd.Series(groupby_obj_train.groups.keys())
filename_series_test = pd.Series(groupby_obj_test.groups.keys())

# Apply the save_data function to each filename in train and test sets
filename_series_train.apply(save_data, args=(train_folder, groupby_obj_train))
filename_series_test.apply(save_data, args=(test_folder, groupby_obj_test))
