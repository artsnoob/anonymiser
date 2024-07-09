import re
from typing import List
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class CustomRecognizer:
    def __init__(self, name, pattern):
        self.name = name
        self.pattern = re.compile(pattern)

    def analyze(self, text: str) -> List[RecognizerResult]:
        return [RecognizerResult(
            entity_type=self.name,
            start=match.start(),
            end=match.end(),
            score=0.85
        ) for match in self.pattern.finditer(text)]

class PhoneNumberRecognizer:
    def __init__(self):
        self.name = "PHONE_NUMBER"
        self.patterns = [
            r'\+31\d{9}\b',  # International format without spaces
            r'\+31\s?\d{2}\s?\d{7}\b',  # International format with optional spaces
            r'0\d{9}\b',  # National format without spaces
            r'0\d{2}[-\s]?\d{7}\b',  # National format with optional separator
            r'0\d{3}[-\s]?\d{6}\b',  # Another national format
            r'\b\d{2}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{2}\b',  # Any 10 digit combination with optional separators
        ]

    def analyze(self, text: str) -> List[RecognizerResult]:
        results = []
        for pattern in self.patterns:
            for match in re.finditer(pattern, text):
                results.append(RecognizerResult(
                    entity_type=self.name,
                    start=match.start(),
                    end=match.end(),
                    score=0.85
                ))
        return results

class PostalCodeRecognizer:
    def __init__(self):
        self.name = "POSTAL_CODE"
        self.pattern = re.compile(r'\b\d{4}\s?[A-Za-z]{2}\b|\b\d{4}-[A-Za-z]{2}\b')

    def analyze(self, text: str) -> List[RecognizerResult]:
        return [RecognizerResult(
            entity_type=self.name,
            start=match.start(),
            end=match.end(),
            score=0.85
        ) for match in self.pattern.finditer(text)]

class CustomAnalyzer:
    def __init__(self):
        self.recognizers = [
            CustomRecognizer("PERSON", r'\b(?!(?:Mijn|Je)\b)(?:[A-Z][a-z]+(?:\s+(?:van|de|der|den|van der|van de|van den))?\s+)*[A-Z][a-z]+\b'),
            CustomRecognizer("EMAIL_ADDRESS", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            CustomRecognizer("ADDRESS", r'\b[A-Z][a-z]+(?:straat|weg|laan|plein|singel|kade|gracht)\s+\d+[a-zA-Z]?\b'),
            PhoneNumberRecognizer(),
            PostalCodeRecognizer()
        ]

    def analyze(self, text: str) -> List[RecognizerResult]:
        results = []
        for recognizer in self.recognizers:
            results.extend(recognizer.analyze(text))
        return sorted(results, key=lambda x: x.start)  # Sort by start index

def anonymize_text(text: str) -> str:
    analyzer = CustomAnalyzer()
    anonymizer = AnonymizerEngine()

    # Analyze the text to find PII entities
    results = analyzer.analyze(text)

    # Print debug information
    print("Detected entities:")
    for result in results:
        print(f"Entity: {result.entity_type}, Start: {result.start}, End: {result.end}, Score: {result.score}")

    # Define anonymization operators for each entity type
    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSOON>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<E-MAIL>"}),
        "ADDRESS": OperatorConfig("replace", {"new_value": "<ADRES>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEFOON>"}),
        "POSTAL_CODE": OperatorConfig("replace", {"new_value": "<POSTCODE>"})
    }

    # Anonymize the text
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )

    # Clean up extra spaces
    anonymized_text = re.sub(r'\s+', ' ', anonymized_result.text)
    
    return anonymized_text.strip()

# Example usage
sample_text = """
Mijn naam is Jan de Vries en ik woon op Kalverstraat 123.
Je kunt me bereiken op jan.devries@voorbeeld.nl of bel me op +31612345678.
Mijn collega Piet van der Berg is bereikbaar op 0687654321.

Jan Doedel,
Godsweg 123
5022GR
6022 gr
06-14436857
06 12456798
010-1234567
010 2145678
"""

anonymized_result = anonymize_text(sample_text)
print("\nGeanonimiseerde tekst:")
print(anonymized_result)
