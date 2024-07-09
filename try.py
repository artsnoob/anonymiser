import re
from typing import List, Tuple
import requests
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class PrecisionImprovedHybridPersonRecognizer:
    def __init__(self):
        self.name = "PERSON"
        self.ollama_url = "http://localhost:11434/api/generate"
        self.name_indicators = [
            r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b(meneer|mevrouw|dokter)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(zei|vertelde|sprak|belt)',
            r'(sprak|praatte|overlegde|belde) met\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bik\s+ben\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bhij\s+heet\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bzij\s+heet\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\bmet\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+gesproken',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+heeft\s+(?:geen|wel)',
            r'\bwaarom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+geen',
            r'\bkan\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+niet',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+kan\s+niet',
        ]

    def analyze(self, text: str) -> List[RecognizerResult]:
        results = []

        # Rule-based recognition using contextual patterns
        for pattern in self.name_indicators:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                name = groups[-1]  # The last group should always be the name
                results.append(RecognizerResult(
                    entity_type=self.name,
                    start=match.start(match.lastindex),
                    end=match.start(match.lastindex) + len(name),
                    score=0.7
                ))

        # Ollama Mistral recognition
        prompt = f"""Analyze the following Dutch text and identify all potential person names, including unconventional, uncommon, or non-Dutch names. Consider the context and sentence structure. Only identify the exact name, not surrounding words.

Respond with a JSON list of objects, where each object has 'name', 'start', and 'end' properties. The 'start' and 'end' should be character indices in the original text. Provide a 'confidence' score between 0 and 1 for each name, based on how certain you are that it's a name.

Text: {text}

Response:"""

        response = requests.post(self.ollama_url, json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })

        if response.status_code == 200:
            try:
                names = eval(response.json()['response'])
                for name in names:
                    results.append(RecognizerResult(
                        entity_type=self.name,
                        start=name['start'],
                        end=name['end'],
                        score=name.get('confidence', 0.8)
                    ))
            except Exception as e:
                print(f"Error parsing Ollama response: {e}")

        # Merge overlapping results
        results.sort(key=lambda x: x.start)
        merged_results = []
        for result in results:
            if not merged_results or result.start > merged_results[-1].end:
                merged_results.append(result)
            else:
                merged_results[-1].end = max(merged_results[-1].end, result.end)
                merged_results[-1].score = max(merged_results[-1].score, result.score)

        return merged_results

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
            r'\+31\d{9}\b',
            r'\+31\s?\d{2}\s?\d{7}\b',
            r'0\d{9}\b',
            r'0\d{2}[-\s]?\d{7}\b',
            r'0\d{3}[-\s]?\d{6}\b',
            r'\b\d{2}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{2}\b',
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
            PrecisionImprovedHybridPersonRecognizer(),
            CustomRecognizer("EMAIL_ADDRESS", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            CustomRecognizer("ADDRESS", r'\b[A-Z][a-z]+(?:straat|weg|laan|plein|singel|kade|gracht)\s+\d+[a-zA-Z]?\b'),
            PhoneNumberRecognizer(),
            PostalCodeRecognizer()
        ]

    def analyze(self, text: str) -> List[RecognizerResult]:
        results = []
        for recognizer in self.recognizers:
            results.extend(recognizer.analyze(text))
        return sorted(results, key=lambda x: x.start)

def preprocess_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

def detect_sensitive_info(text: str) -> List[Tuple[str, int, int]]:
    analyzer = CustomAnalyzer()
    return [(result.entity_type, result.start, result.end) for result in analyzer.analyze(text)]

def anonymize_text(text: str) -> str:
    preprocessed_text = preprocess_text(text)
    entities = detect_sensitive_info(preprocessed_text)
    
    anonymizer = AnonymizerEngine()
    
    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSOON>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<E-MAIL>"}),
        "ADDRESS": OperatorConfig("replace", {"new_value": "<ADRES>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TELEFOON>"}),
        "POSTAL_CODE": OperatorConfig("replace", {"new_value": "<POSTCODE>"})
    }
    
    analyzer_results = [RecognizerResult(
        entity_type=entity_type,
        start=start,
        end=end,
        score=0.85
    ) for entity_type, start, end in entities]
    
    anonymized_result = anonymizer.anonymize(
        text=preprocessed_text,
        analyzer_results=analyzer_results,
        operators=operators
    )
    
    return anonymized_result.text

def main():
    print("Welcome to the Precision Improved Text Anonymization Tool!")
    print("Enter your text below. Press Enter twice to finish input.")
    
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    
    input_text = "\n".join(lines)
    
    print("\nOriginal text:")
    print(input_text)
    
    anonymized_result = anonymize_text(input_text)
    
    print("\nAnonymized text:")
    print(anonymized_result)

if __name__ == "__main__":
    main()
