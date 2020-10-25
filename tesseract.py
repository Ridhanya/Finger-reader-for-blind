from PIL import Image
import pytesseract
im= Image.open('fr.jpg')
text = pytesseract.image_to_string(im)
print(text)
