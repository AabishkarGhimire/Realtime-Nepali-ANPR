from PIL import Image
import pytesseract

# Replace 'image_path' with the path to your Nepali text image
image_path = './a.png'

# Set the language to 'nep' for Nepali
result = pytesseract.image_to_string(Image.open(image_path), lang='nep')

print(result)
