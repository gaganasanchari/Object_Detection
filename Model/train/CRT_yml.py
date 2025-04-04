# Read classes from classes.txt
with open("./classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Generate data.yaml content
data_yaml = f"""
train: /home/ganesh/Videos/Jayesh_CV/Code_30/Model/train/images/train/
val: /home/ganesh/Videos/Jayesh_CV/Code_30/Model/train/images/val/
nc: {len(class_names)}
names: {class_names}
"""

# Save to data.yaml
with open("data.yaml", "w") as f:
    f.write(data_yaml)
