import json
import random
from sklearn.model_selection import train_test_split

# Read the JSON file and extract the image IDs
with open('visual_genome_relation.json') as f:
    data = json.load(f)
    image_ids = [item['image_id'] for item in data]

# Get the unique image IDs
unique_image_ids = set(image_ids)

# Shuffle the image IDs
shuffled_image_ids = list(unique_image_ids)
random.shuffle(shuffled_image_ids)

# Split the data into train, test, and validation sets
train_ids, test_ids = train_test_split(shuffled_image_ids, test_size=0.2)
test_ids, val_ids = train_test_split(test_ids, test_size=0.5)

# Create empty lists for train, test, and validation data
train_data = []
test_data = []
val_data = []

# Iterate over the image IDs and put the items in the corresponding lists
for item in data:
    if item['image_id'] in train_ids:
        train_data.append(item)
    elif item['image_id'] in test_ids:
        test_data.append(item)
    elif item['image_id'] in val_ids:
        val_data.append(item)

# Save the lists in a JSON file
with open('train_visual_genome_relation.json', 'w') as f:
    json.dump(train_data, f)

with open('val_visual_genome_relation.json', 'w') as f:
    json.dump(test_data, f)

with open('test_visual_genome_relation.json', 'w') as f:
    json.dump(val_data, f)

# Print the results
print("total data:", len(data)) # 23937
print("Train Data:", len(train_data)) # 18906
print("Test Data:", len(test_data)) # 2449
print("Validation Data:", len(val_data)) #2582

print("total data:", len(unique_image_ids)) # 5316
print("Train Data:", len(train_ids)) # 4252
print("Test Data:", len(test_ids)) # 532
print("Validation Data:", len(val_ids)) # 532

