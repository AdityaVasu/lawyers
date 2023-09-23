from PIL import Image
import io

# Open an image file
image_path = 'path_to_your_image.jpg'
with open(image_path, 'rb') as image_file:
    image = Image.open(image_file)

# Convert the image to bytes
image_bytes = io.BytesIO()
image.save(image_bytes, format='JPEG')  # You can specify the format you want (JPEG, PNG, etc.)
image_bytes = image_bytes.getvalue()