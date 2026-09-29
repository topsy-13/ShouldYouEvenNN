import openml
import json

def export_openml_dict(dataset_ids, output_file="openml_datasets2.json"):
    dataset_dict = {}

    for ds_id in dataset_ids:
        try:
            dataset = openml.datasets.get_dataset(ds_id)
            dataset_dict[dataset.name] = ds_id
        except Exception as e:
            print(f"Error fetching dataset {ds_id}: {e}")

    # Sort by dataset name
    sorted_dict = dict(sorted(dataset_dict.items(), key=lambda x: x[0].lower()))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, indent=4, ensure_ascii=False)

    print(f"Exported {len(sorted_dict)} datasets to {output_file}")

# Usage
dataset_ids1 = [41138, 4135, 40981, 40996, 1111, 41150, 1590, 1169, 41147, 1461, 1464, 40975, 41142, 1468, 40668, 1596, 31, 41167, 41164, 41169, 23512, 41168, 41143, 41027, 1067, 3, 12, 1486, 23517, 1489, 40984, 40685, 41146, 54, 41166]
dataset_ids2 = [3, 12, 23, 31, 54, 181, 188, 1049, 1067, 1111 , 1169, 1457, 1461, 1464, 1468, 1475, 1486, 1487, 1489, 1494, 1515, 1590, 1596, 4134, 4135, 4534, 4538, 4541, 23517, 40498, 40668, 40670, 40685, 40701, 40900, 40975, 40978, 40981, 40982, 40983, 40984, 40996, 41027, 41138, 41142, 41143, 41144, 41145, 41146, 41147, 41150, 41156, 41157, 41158, 41159, 41161, 41162, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 42732, 42733, 42734, 42742, 42746, 42769, 43072]

# Mix both lists and remove duplicates
dataset_ids = list(set(dataset_ids1 + dataset_ids2))
# Print common elements in both list and differences lenght
print(f"Total unique datasets: {len(dataset_ids)}")
print(f"Common datasets in both lists: {len(set(dataset_ids1).intersection(set(dataset_ids2)))}")
print(f"Datasets only in first list: {len(set(dataset_ids1) - set(dataset_ids2))}")
print(set(dataset_ids1) - set(dataset_ids2))

print(f"Datasets only in second list: {len(set(dataset_ids2) - set(dataset_ids1))}")

export_openml_dict(dataset_ids)