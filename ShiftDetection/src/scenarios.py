from torch.utils.data import Dataset, Subset, random_split


def split_by_class(dataset, groups):
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    indices_per_group = [[] for _ in groups]
    for point_idx, y in enumerate(labels):
        for group_idx, group in enumerate(groups):
            if y in group:
                indices_per_group[group_idx].append(point_idx)
                break
    return [Subset(dataset, indices) for indices in indices_per_group]


class BinarizedDataset(Dataset):

    def __init__(self, dataset, flip=False):
        self._dataset = dataset
        self._flip = flip

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        x, y = item[0], int(item[1])
        y = y % 2
        if self._flip:
            y = 1 - y
        return x, y


def make_binary_superclass_scenario(dataset, reference_size):
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    num_classes = max(labels) + 1
    num_reference_classes = round(0.8 * num_classes)
    reference_classes = list(range(num_reference_classes))
    out_of_support_classes = list(range(num_reference_classes, num_classes))
    in_support_subset_classes = list(range(round(0.5 * num_reference_classes)))

    ds_ref, ds_out_of_support_covshift = split_by_class(dataset, [reference_classes, out_of_support_classes])
    ds_ref, ds_in_support = random_split(ds_ref, [reference_size, len(ds_ref) - reference_size])

    lengths = [round(0.25*len(ds_in_support))] * 3
    lengths.append(len(ds_in_support) - sum(lengths))
    ds_in_distribution, ds_in_support_covshift, ds_pure_concept_shift, ds_combined_shift = random_split(
        ds_in_support, lengths)
    ds_in_support_covshift = split_by_class(ds_in_support_covshift, [in_support_subset_classes])[0]
    ds_combined_shift = split_by_class(ds_combined_shift, [in_support_subset_classes])[0]

    ds_ref = BinarizedDataset(ds_ref)
    query_sets = {
        "in_distribution": BinarizedDataset(ds_in_distribution),
        "in_support_covariate_shift": BinarizedDataset(ds_in_support_covshift),
        "out_of_support_covariate_shift": BinarizedDataset(ds_out_of_support_covshift),
        "pure_concept_shift": BinarizedDataset(ds_pure_concept_shift, flip=True),
        "combined_shift": BinarizedDataset(ds_combined_shift, flip=True),
    }
    return ds_ref, query_sets
