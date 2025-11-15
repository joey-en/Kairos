import os
from dotenv import load_dotenv
load_dotenv(".env")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

# ========================================================

from google import genai
from google.genai import types

with open(r'output\frames\scene_006\frame_01.jpg', 'rb') as f:
    image_bytes = f.read()

client = genai.Client(vertexai=True, api_key=api_key)
response = client.models.generate_content(
model='gemini-2.5-pro',
contents=[
    types.Part.from_bytes(
    data=image_bytes,
    mime_type='image/jpeg',
    ),
    '''
    Before this scene: a video frame of a cartoon character sitting at a table
    Now: a video frame of a sponge sponge with a piece of paper
    After this scene: a video frame of a sponge sponge with a piece of paper

    Based on the image and this context, concisely describe what is happening in this frame, focusing on new details or clarifications
    '''
]
)
print(response.text)
# Scene 0
#   a video frame of a room with a blue door and a pink flower on the floor
#   a video frame of sponge spongenan ' s revenge
#   a video frame of a sponge sponge and his friend
# Scene 1
#   a video frame of sponge spongenan ' s revenge
#   a video frame of sponge spongenan ' s revenge
#   a video frame of sponge spongenan ' s revenge
# Scene 2
#   a video frame of blue and green stripes
#   a video frame of a cartoon character holding a sword
#   a video frame of a man with a hat on his head
# Scene 3
#   a video frame of a piece of paper on a table
#   a video frame of a person writing on a piece of paper
#   a video frame of a pencil on a piece of paper
# Scene 4
#   a video frame of a cartoon character sitting on a chair
#   a video frame of a sponge sponge and his friend
#   a video frame of a cartoon character with blue eyes and a smile
# Scene 5
#   a video frame of an airplane flying in the sky
#   a video frame of a dog holding a pencil
#   a video frame of a cartoon character holding a pencil
# Scene 6
#   a video frame of a cartoon character sitting at a table
#   a video frame of a sponge sponge with a piece of paper
#   a video frame of a sponge sponge with a piece of paper
# Scene 7
#   a video frame of a cartoon character holding a piece of paper
#   a video frame of a cartoon character holding a piece of paper
#   a video frame of a person holding a piece of paper
# Scene 8
#   a video frame of a sponge sponge and his friend
#   a video frame of a sponge sponge and his friend
#   a video frame of a sponge sponge and his friend
# Scene 9
#   a video frame of a black background with a white border