import re
from num2words import num2words

def normalize_numbers_spanish(text):
    """
    Normalize numbers in text to Spanish words.
    
    Args:
        text (str): Input text containing numbers
        
    Returns:
        str: Text with numbers converted to Spanish words
    """
    # Find all numbers in the text (including decimals)
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    
    for num in numbers:
        try:
            # Convert to float if decimal
            num_float = float(num)
            # Convert to Spanish words
            word = num2words(num_float, lang='es')
            # Replace in text
            text = text.replace(num, word)
        except ValueError:
            # Skip if conversion fails
            continue
    
    return text 