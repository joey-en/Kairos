
def see_first_scene(df):
    print("Printing first captioned scene:")
    print("{")
    for key in df[0]:
        if key == "frames": continue
        print(f"{key}, {df[0][key]},")
    print("}")

def see_scenes_cuts(df):
    print(f"Found {len(df)} scenes.")
    for s in df:
        print(
            f"Scene {s['scene_index']:03d}: "
            f"{s['start_timecode']} -> {s['end_timecode']} "
            f"({s['duration_seconds']:.2f} sec)"
        )

def save_scenes_to_file(df, filepath):
    import json
    with open(filepath, 'w') as f:
        json.dump(df, f, indent=4)
    print(f"Scenes saved to {filepath}")