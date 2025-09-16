#!/usr/bin/env python3
"""
Synthetic Data Generator for NLP

This script generates synthetic text data for Natural Language Processing tasks
in multiple languages (English, Japanese, Vietnamese) with different dataset sizes.

Author: NLP Learning Journey
License: MIT
"""

import os
import random
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json


class SyntheticTextGenerator:
    """Generates synthetic text data for NLP tasks in multiple languages."""
    
    def __init__(self):
        """Initialize the synthetic text generator with language-specific templates."""
        
        # English templates and vocabulary
        self.english_templates = {
            'news': [
                "Today, {location} announced {action} regarding {topic}.",
                "In a recent development, {organization} has {action} {object}.",
                "The {adjective} {event} took place in {location} yesterday.",
                "According to {source}, {statement} about {topic}.",
                "Local authorities in {location} reported {event} involving {subject}."
            ],
            'reviews': [
                "This {product} is {adjective}. I {feeling} it because {reason}.",
                "I {action} {product} and found it to be {adjective}.",
                "The {service} was {adjective}, especially the {aspect}.",
                "Would {recommendation} this {product} to {target_audience}.",
                "Overall experience was {adjective} due to {reason}."
            ],
            'social': [
                "Just {action} at {location} and it was {adjective}!",
                "Can't believe {event} happened today. So {feeling}!",
                "Looking forward to {future_event} next {time_period}.",
                "Thanks to {person} for {action}. Really {feeling}!",
                "Anyone know about {topic}? Need {help_type}."
            ]
        }
        
        self.english_vocab = {
            'location': ['New York', 'London', 'Tokyo', 'Sydney', 'Berlin', 'Paris', 'Toronto', 'Mumbai'],
            'action': ['announced', 'revealed', 'disclosed', 'confirmed', 'reported', 'stated', 'declared'],
            'topic': ['climate change', 'technology', 'education', 'healthcare', 'economy', 'sports', 'culture'],
            'organization': ['the government', 'Microsoft', 'Google', 'the university', 'the company', 'NASA'],
            'adjective': ['amazing', 'terrible', 'good', 'bad', 'excellent', 'poor', 'outstanding', 'disappointing'],
            'event': ['conference', 'meeting', 'celebration', 'protest', 'festival', 'competition', 'exhibition'],
            'source': ['experts', 'officials', 'researchers', 'analysts', 'witnesses', 'studies'],
            'statement': ['new findings', 'significant progress', 'important changes', 'major developments'],
            'subject': ['students', 'residents', 'employees', 'visitors', 'participants', 'citizens'],
            'product': ['smartphone', 'laptop', 'restaurant', 'movie', 'book', 'service', 'hotel', 'app'],
            'feeling': ['love', 'hate', 'like', 'enjoy', 'appreciate', 'recommend', 'prefer'],
            'reason': ['the quality', 'the price', 'the service', 'the design', 'the features', 'the experience'],
            'service': ['customer service', 'delivery', 'support', 'interface', 'performance', 'reliability'],
            'aspect': ['staff', 'atmosphere', 'quality', 'speed', 'convenience', 'value'],
            'recommendation': ['recommend', 'suggest', 'advise', 'endorse', 'propose'],
            'target_audience': ['everyone', 'friends', 'family', 'colleagues', 'travelers', 'students'],
            'future_event': ['vacation', 'meeting', 'conference', 'party', 'event', 'celebration'],
            'time_period': ['week', 'month', 'year', 'summer', 'winter', 'weekend'],
            'person': ['everyone', 'my friend', 'my colleague', 'the team', 'my family', 'the organizers'],
            'help_type': ['advice', 'information', 'assistance', 'suggestions', 'recommendations', 'help'],
            'object': ['new policies', 'updates', 'improvements', 'changes', 'initiatives', 'programs']
        }
        
        # Japanese templates (romanized for compatibility)
        self.japanese_templates = [
            "Kyou wa {weather} desu. {activity} wo shimashita.",
            "{place} de {event} ga arimashita. Totemo {adjective} deshita.",
            "{person} san wa {action} shimashita. {feeling} desu.",
            "{time} ni {location} de {meeting} ga arimasu.",
            "{product} wo {action} mashita. {opinion} to omoimasu.",
            "{season} ni natte, {activity} ga {adjective} ni narimashita.",
            "{company} no {service} wa {quality} desu.",
            "Ashita {plan} desu. {feeling} shite imasu."
        ]
        
        self.japanese_vocab = {
            'weather': ['ii tenki', 'ame', 'yuki', 'kumori', 'hare'],
            'activity': ['benkyou', 'shigoto', 'ryokou', 'kaimono', 'undou'],
            'place': ['Tokyo', 'Osaka', 'Kyoto', 'Yokohama', 'Kobe'],
            'event': ['matsuri', 'kaigi', 'paatii', 'konsaato', 'tenrankai'],
            'adjective': ['tanoshii', 'tsumaranai', 'muzukashii', 'yasashii', 'oishii'],
            'person': ['tomodachi', 'kazoku', 'sensei', 'dooryou', 'senpai'],
            'action': ['taberu', 'nomu', 'iku', 'kuru', 'miru'],
            'feeling': ['ureshii', 'kanashii', 'tanoshii', 'shinpai', 'anshin'],
            'time': ['asa', 'hiru', 'yoru', 'ashita', 'kyou'],
            'location': ['eki', 'gakkou', 'kaisha', 'ie', 'mise'],
            'meeting': ['kaigi', 'awakai', 'mendan', 'uchiawase', 'hanashiai'],
            'product': ['hon', 'eiga', 'geemu', 'kuruma', 'keitai'],
            'opinion': ['ii', 'warui', 'futsuu', 'suki', 'kirai'],
            'season': ['haru', 'natsu', 'aki', 'fuyu'],
            'company': ['kaisha', 'mise', 'gakkou', 'byouin', 'ginkou'],
            'service': ['saabisu', 'shina', 'ryouri', 'shigoto', 'tetsudai'],
            'quality': ['ii', 'warui', 'futsuu', 'subarashii', 'zannen'],
            'plan': ['ryokou', 'kaigi', 'paatii', 'benkyou', 'yasumi']
        }
        
        # Vietnamese templates
        self.vietnamese_templates = [
            "Hôm nay tôi đã {action} ở {place}. Rất {adjective}!",
            "{person} đã {action} {object}. Tôi cảm thấy {feeling}.",
            "Ở {location} có {event} rất {adjective}.",
            "{time} này, {weather} {adjective} quá.",
            "Tôi {feeling} {activity} vào {time}.",
            "{product} này {quality}. Tôi {recommendation}.",
            "Gia đình tôi {action} đến {place} vào {time}.",
            "{event} ở {location} sẽ {action} vào {time}."
        ]
        
        self.vietnamese_vocab = {
            'action': ['đi', 'ăn', 'học', 'làm việc', 'chơi', 'xem', 'mua', 'bán'],
            'place': ['nhà hàng', 'trường học', 'công viên', 'chợ', 'bệnh viện', 'ngân hàng'],
            'adjective': ['tốt', 'xấu', 'đẹp', 'hay', 'dở', 'thú vị', 'nhàm chán', 'tuyệt vời'],
            'person': ['bạn tôi', 'gia đình', 'đồng nghiệp', 'thầy cô', 'hàng xóm'],
            'object': ['sách', 'điện thoại', 'xe', 'nhà', 'quần áo', 'đồ ăn'],
            'feeling': ['vui', 'buồn', 'hạnh phúc', 'lo lắng', 'thích', 'ghét'],
            'location': ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ'],
            'event': ['lễ hội', 'họp mặt', 'tiệc', 'hội nghị', 'triển lãm', 'buổi hòa nhạc'],
            'time': ['sáng', 'trưa', 'chiều', 'tối', 'ngày mai', 'tuần tới'],
            'weather': ['thời tiết', 'trời', 'nắng', 'mưa', 'gió'],
            'activity': ['đọc sách', 'xem phim', 'nghe nhạc', 'chơi game', 'thể thao'],
            'product': ['sản phẩm', 'dịch vụ', 'món ăn', 'đồ uống', 'quần áo'],
            'quality': ['tốt', 'không tốt', 'bình thường', 'xuất sắc', 'tệ'],
            'recommendation': ['khuyên dùng', 'không khuyên', 'thích', 'không thích']
        }
    
    def generate_english_sentence(self) -> str:
        """Generate a single English sentence using templates and vocabulary."""
        category = random.choice(list(self.english_templates.keys()))
        template = random.choice(self.english_templates[category])
        
        # Fill in the template with random vocabulary
        sentence = template
        for placeholder in self.english_vocab:
            if '{' + placeholder + '}' in sentence:
                replacement = random.choice(self.english_vocab[placeholder])
                sentence = sentence.replace('{' + placeholder + '}', replacement)
        
        return sentence
    
    def generate_japanese_sentence(self) -> str:
        """Generate a single Japanese sentence using templates and vocabulary."""
        template = random.choice(self.japanese_templates)
        
        # Fill in the template with random vocabulary
        sentence = template
        for placeholder in self.japanese_vocab:
            if '{' + placeholder + '}' in sentence:
                replacement = random.choice(self.japanese_vocab[placeholder])
                sentence = sentence.replace('{' + placeholder + '}', replacement)
        
        return sentence
    
    def generate_vietnamese_sentence(self) -> str:
        """Generate a single Vietnamese sentence using templates and vocabulary."""
        template = random.choice(self.vietnamese_templates)
        
        # Fill in the template with random vocabulary
        sentence = template
        for placeholder in self.vietnamese_vocab:
            if '{' + placeholder + '}' in sentence:
                replacement = random.choice(self.vietnamese_vocab[placeholder])
                sentence = sentence.replace('{' + placeholder + '}', replacement)
        
        return sentence
    
    def generate_text(self, language: str, word_count: int) -> str:
        """Generate text with approximately the specified number of words."""
        sentences = []
        current_word_count = 0
        
        # Choose the appropriate sentence generator
        if language == 'english':
            generator_func = self.generate_english_sentence
        elif language == 'japanese':
            generator_func = self.generate_japanese_sentence
        elif language == 'vietnamese':
            generator_func = self.generate_vietnamese_sentence
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        # Generate sentences until we reach the target word count
        while current_word_count < word_count:
            sentence = generator_func()
            sentences.append(sentence)
            # Approximate word count (split by spaces, though this is rough for Japanese)
            current_word_count += len(sentence.split())
        
        return '\n'.join(sentences)
    
    def save_data(self, text: str, filepath: str) -> None:
        """Save generated text to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Generated data saved to: {filepath}")
    
    def generate_all_datasets(self, base_dir: str = '../data') -> None:
        """Generate all required datasets for all languages and sizes."""
        languages = ['english', 'japanese', 'vietnamese']
        sizes = {
            'small': 1000,
            'medium': 100000,
            'large': 1000000
        }
        
        for language in languages:
            print(f"\nGenerating {language} datasets...")
            for size_name, word_count in sizes.items():
                print(f"  Generating {size_name} dataset ({word_count:,} words)...")
                text = self.generate_text(language, word_count)
                filepath = os.path.join(base_dir, language, f"{size_name}.txt")
                self.save_data(text, filepath)
        
        print(f"\nAll datasets generated successfully!")
        self.print_summary(base_dir)
    
    def print_summary(self, base_dir: str) -> None:
        """Print a summary of generated files."""
        print("\n" + "="*50)
        print("GENERATION SUMMARY")
        print("="*50)
        
        languages = ['english', 'japanese', 'vietnamese']
        sizes = ['small', 'medium', 'large']
        
        for language in languages:
            print(f"\n{language.capitalize()} files:")
            for size in sizes:
                filepath = os.path.join(base_dir, language, f"{size}.txt")
                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        word_count = len(content.split())
                        line_count = len(content.split('\n'))
                    
                    print(f"  {size}.txt: {file_size:,} bytes, {word_count:,} words, {line_count:,} lines")


def main():
    """Main function to handle command-line arguments and execute data generation."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic text data for NLP tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python synthetic-data.py                    # Generate all datasets
  python synthetic-data.py --language english --size small  # Generate specific dataset
  python synthetic-data.py --output /custom/path            # Custom output directory
        """
    )
    
    parser.add_argument(
        '--language',
        choices=['english', 'japanese', 'vietnamese'],
        help='Generate data for specific language only'
    )
    
    parser.add_argument(
        '--size',
        choices=['small', 'medium', 'large'],
        help='Generate data for specific size only (requires --language)'
    )
    
    parser.add_argument(
        '--output',
        default='../data',
        help='Output directory for generated data (default: ../data)'
    )
    
    parser.add_argument(
        '--words',
        type=int,
        help='Custom word count (requires --language)'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SyntheticTextGenerator()
    
    # Validate arguments
    if args.size and not args.language:
        print("Error: --size requires --language to be specified")
        sys.exit(1)
    
    if args.words and not args.language:
        print("Error: --words requires --language to be specified")
        sys.exit(1)
    
    # Generate data based on arguments
    if args.language:
        if args.words:
            # Custom word count
            print(f"Generating {args.language} text with {args.words:,} words...")
            text = generator.generate_text(args.language, args.words)
            filename = f"custom_{args.words}.txt"
            filepath = os.path.join(args.output, args.language, filename)
            generator.save_data(text, filepath)
        elif args.size:
            # Specific language and size
            size_map = {'small': 1000, 'medium': 100000, 'large': 1000000}
            word_count = size_map[args.size]
            print(f"Generating {args.language} {args.size} dataset ({word_count:,} words)...")
            text = generator.generate_text(args.language, word_count)
            filepath = os.path.join(args.output, args.language, f"{args.size}.txt")
            generator.save_data(text, filepath)
        else:
            # All sizes for specific language
            print(f"Generating all datasets for {args.language}...")
            sizes = {'small': 1000, 'medium': 100000, 'large': 1000000}
            for size_name, word_count in sizes.items():
                print(f"  Generating {size_name} dataset ({word_count:,} words)...")
                text = generator.generate_text(args.language, word_count)
                filepath = os.path.join(args.output, args.language, f"{size_name}.txt")
                generator.save_data(text, filepath)
    else:
        # Generate all datasets
        print("Generating all datasets for all languages...")
        generator.generate_all_datasets(args.output)


if __name__ == "__main__":
    main()