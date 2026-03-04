def omr_collate_fn(batch):
    """
    Groups variable-sized page data without forcing them into strict multi-dimensional tensors.
    """
    images = []
    boxes_list = []
    labels_list = []
    edges_list = []
    file_names = []

    for item in batch:
        images.append(item["image"])
        boxes_list.append(item["boxes"])
        labels_list.append(item["labels"])
        edges_list.append(item["edges"])
        file_names.append(item["file_name"])

    # We can stack the images if they are padded to the exact same size,
    # but for full pages it is often safer to just return the list.
    # Assuming your model handles a list of image tensors:

    return {
        "images": images,  # List of (3, H, W) tensors
        "boxes": boxes_list,  # List of (N, 4) tensors
        "labels": labels_list,  # List of (N) tensors
        "edges": edges_list,  # List of (E, 3) tensors
        "file_names": file_names,
    }


# --- Usage Example ---
# class_list = config["target_classes"]
# dataset = OMRFullPageDataset("data/output_imgs", "data/output_annotations", "data/output_graphs", class_list)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=omr_collate_fn)
