def fuse_blip_yolo(blip_caption, yolo_objects):
    """
    Combine BLIP caption + YOLO objects into a single text prompt.
    """
    object_list = ", ".join([obj["label"] for obj in yolo_objects])
    return (
        f"BLIP Caption: {blip_caption}\n"
        f"Detected Objects: {object_list}\n"
        f"Combined Understanding: The scene likely contains {object_list} "
        f"and visually looks like: {blip_caption}."
    )
