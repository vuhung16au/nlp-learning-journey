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
        
        # Japanese templates (using proper Japanese characters)
        self.japanese_templates = [
            "今日は{weather}です。{activity}をしました。",
            "{place}で{event}がありました。とても{adjective}でした。",
            "{person}さんは{action}しました。{feeling}です。",
            "{time}に{location}で{meeting}があります。",
            "{product}を{action}ました。{opinion}と思います。",
            "{season}になって、{activity}が{adjective}になりました。",
            "{company}の{service}は{quality}です。",
            "明日{plan}です。{feeling}しています。"
        ]
        
        self.japanese_vocab = {
            'weather': ['いい天気', '雨', '雪', '曇り', '晴れ'],
            'activity': ['勉強', '仕事', '旅行', '買い物', '運動'],
            'place': ['東京', '大阪', '京都', '横浜', '神戸'],
            'event': ['祭り', '会議', 'パーティー', 'コンサート', '展覧会'],
            'adjective': ['楽しい', 'つまらない', '難しい', '優しい', '美味しい'],
            'person': ['友達', '家族', '先生', '同僚', '先輩'],
            'action': ['食べ', '飲み', '行き', '来', '見'],
            'feeling': ['嬉しい', '悲しい', '楽しい', '心配', '安心'],
            'time': ['朝', '昼', '夜', '明日', '今日'],
            'location': ['駅', '学校', '会社', '家', '店'],
            'meeting': ['会議', '打ち合わせ', '面談', '打合せ', '話し合い'],
            'product': ['本', '映画', 'ゲーム', '車', '携帯'],
            'opinion': ['いい', '悪い', '普通', '好き', '嫌い'],
            'season': ['春', '夏', '秋', '冬'],
            'company': ['会社', '店', '学校', '病院', '銀行'],
            'service': ['サービス', '品', '料理', '仕事', '手伝い'],
            'quality': ['いい', '悪い', '普通', '素晴らしい', '残念'],
            'plan': ['旅行', '会議', 'パーティー', '勉強', '休み']
        }
        
        # Vietnamese templates - improved with authentic language patterns
        self.vietnamese_templates = [
            # Personal introductions and greetings (following repository examples)
            "Tên tôi là {name}. Tôi đến từ {location}.",
            "Xin chào! Tôi là {profession} tại {workplace}.",
            "Rất vui được gặp {person}. Tôi làm việc ở {workplace}.",
            
            # Daily activities and experiences
            "Hôm nay tôi đã {action} ở {place}. Cảm thấy rất {feeling}!",
            "Tôi thường {activity} vào {time_period}.",
            "Cuối tuần tôi thích {leisure_activity} với {companion}.",
            
            # Opinions and reviews
            "{product} này thật {quality}. Tôi {recommendation} cho mọi người.",
            "Dịch vụ tại {business} rất {service_quality}. {opinion_detail}.",
            "Món {food} có vị {taste} và được chế biến rất {cooking_quality}.",
            
            # Events and experiences
            "{event} ở {location} diễn ra rất {event_quality}.",
            "Tuần trước tôi tham gia {activity} tại {venue}.",
            "{person} đã {action} một {achievement} đáng {appreciation}.",
            
            # Weather and environment
            "Hôm nay thời tiết {weather_condition}, thích hợp để {weather_activity}.",
            "Mùa {season} ở {location} rất {seasonal_quality}.",
            
            # Future plans
            "Tuần tới tôi sẽ {future_action} ở {destination}.",
            "Tôi đang lên kế hoạch {plan} vào {future_time}.",
            
            # Technology and learning (following repository's programming focus)
            "Tôi đang học {subject} để {learning_goal}.",
            "Công nghệ {technology} giúp {benefit} trong {field}.",
            "Lập trình {programming_language} rất {learning_difficulty} nhưng {reward}.",
            
            # Social interactions
            "{person} đã chia sẻ với tôi về {topic}. Thật {reaction}!",
            "Gia đình tôi {family_activity} vào {family_time}.",
            "Bạn bè tôi thường {social_activity} tại {social_venue}."
        ]
        
        self.vietnamese_vocab = {
            # Names (common Vietnamese names)
            'name': ['Minh', 'Hoa', 'Nam', 'Linh', 'Dung', 'Tú', 'An', 'Bình', 'Chi', 'Long'],
            
            # Locations (Vietnamese cities and places)
            'location': ['Hà Nội', 'Hồ Chí Minh', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ', 'Hải Dương', 'Vinh', 'Huế'],
            
            # Professions
            'profession': ['kỹ sư', 'giáo viên', 'bác sĩ', 'lập trình viên', 'sinh viên', 'nhân viên', 'thầy thuốc', 'kiến trúc sư'],
            
            # Workplaces
            'workplace': ['công ty', 'trường đại học', 'bệnh viện', 'ngân hàng', 'cửa hàng', 'nhà máy', 'văn phòng', 'trung tâm'],
            
            # People relationships
            'person': ['bạn', 'gia đình', 'đồng nghiệp', 'thầy cô', 'anh chị', 'bố mẹ', 'người bạn', 'mọi người'],
            
            # Actions (verbs)
            'action': ['đi', 'ăn', 'học', 'làm việc', 'chơi', 'xem', 'mua', 'đọc', 'viết', 'nấu ăn', 'tập thể dục'],
            
            # Places
            'place': ['nhà hàng', 'trường học', 'công viên', 'chợ', 'thư viện', 'bảo tàng', 'rạp chiếu phim', 'quán cà phê'],
            
            # Feelings and emotions
            'feeling': ['vui', 'hạnh phúc', 'thú vị', 'hài lòng', 'thoải mái', 'tự tin', 'biết ơn', 'phấn khích'],
            
            # Activities
            'activity': ['đọc sách', 'xem phim', 'nghe nhạc', 'chơi game', 'tập thể thao', 'nấu ăn', 'vẽ tranh', 'chụp ảnh'],
            
            # Time periods
            'time_period': ['buổi sáng', 'buổi trưa', 'buổi chiều', 'buổi tối', 'cuối tuần', 'ngày nghỉ', 'kỳ nghỉ'],
            
            # Leisure activities
            'leisure_activity': ['đi dạo', 'đi du lịch', 'đi ăn', 'xem phim', 'chơi thể thao', 'đi mua sắm', 'thăm bạn bè'],
            
            # Companions
            'companion': ['gia đình', 'bạn bè', 'đồng nghiệp', 'người yêu', 'anh chị em', 'bố mẹ'],
            
            # Products
            'product': ['điện thoại', 'máy tính', 'sách', 'phim', 'ứng dụng', 'website', 'trò chơi', 'xe hơi'],
            
            # Quality descriptors
            'quality': ['tuyệt vời', 'rất tốt', 'ổn', 'không tốt', 'xuất sắc', 'chất lượng cao', 'đáng tin cậy'],
            
            # Recommendations
            'recommendation': ['rất khuyên', 'khuyên', 'không khuyên', 'giới thiệu', 'đề xuất'],
            
            # Business types
            'business': ['nhà hàng', 'khách sạn', 'cửa hàng', 'siêu thị', 'quán cà phê', 'salon tóc', 'spa'],
            
            # Service quality
            'service_quality': ['tốt', 'xuất sắc', 'chuyên nghiệp', 'thân thiện', 'nhanh chóng', 'chu đáo'],
            
            # Opinion details
            'opinion_detail': ['Nhân viên rất thân thiện', 'Giá cả hợp lý', 'Không gian thoải mái', 'Chất lượng đảm bảo'],
            
            # Food items
            'food': ['phở', 'bún bò Huế', 'bánh mì', 'cơm tấm', 'chả cá', 'nem rán', 'bánh cuốn'],
            
            # Taste descriptors
            'taste': ['ngon', 'đậm đà', 'thanh mát', 'cay', 'ngọt', 'chua', 'mặn vừa'],
            
            # Cooking quality
            'cooking_quality': ['tốt', 'cẩn thận', 'khéo léo', 'chuyên nghiệp', 'truyền thống'],
            
            # Events
            'event': ['hội chợ', 'triển lãm', 'buổi hòa nhạc', 'lễ hội', 'hội thảo', 'cuộc thi', 'sự kiện'],
            
            # Event quality
            'event_quality': ['thành công', 'sôi động', 'ý nghĩa', 'bổ ích', 'thú vị', 'ấn tượng'],
            
            # Venues
            'venue': ['trung tâm hội nghị', 'nhà văn hóa', 'sân vận động', 'công viên', 'bảo tàng', 'rạp chiếu phim'],
            
            # Achievements
            'achievement': ['thành tích', 'dự án', 'ý tưởng', 'kết quả', 'sáng kiến', 'giải pháp'],
            
            # Appreciation
            'appreciation': ['ngưỡng mộ', 'trân trọng', 'tự hào', 'ấn tượng', 'đánh giá cao'],
            
            # Weather conditions
            'weather_condition': ['đẹp', 'mát mẻ', 'nắng ấm', 'mưa nhẹ', 'trong lành', 'dễ chịu'],
            
            # Weather activities
            'weather_activity': ['đi dạo', 'tập thể dục', 'đi picnic', 'chụp ảnh', 'đi du lịch'],
            
            # Seasons
            'season': ['xuân', 'hạ', 'thu', 'đông', 'mùa mưa', 'mùa khô'],
            
            # Seasonal quality
            'seasonal_quality': ['đẹp', 'dễ chịu', 'thú vị', 'tuyệt vời', 'lãng mạn', 'trong lành'],
            
            # Future actions
            'future_action': ['đi du lịch', 'tham gia hội thảo', 'gặp gỡ bạn bè', 'học tập', 'làm việc'],
            
            # Destinations
            'destination': ['Hà Nội', 'Sài Gòn', 'Đà Lạt', 'Phú Quốc', 'Hạ Long', 'Sapa', 'Nha Trang'],
            
            # Plans
            'plan': ['du lịch', 'học tập', 'làm việc', 'khởi nghiệp', 'phát triển kỹ năng'],
            
            # Future time
            'future_time': ['tháng tới', 'năm sau', 'mùa hè', 'cuối năm', 'kỳ nghỉ'],
            
            # Subjects (academic/professional)
            'subject': ['lập trình', 'tiếng Anh', 'kinh doanh', 'marketing', 'thiết kế', 'kỹ thuật', 'y học'],
            
            # Learning goals
            'learning_goal': ['phát triển sự nghiệp', 'nâng cao kỹ năng', 'tìm việc làm tốt', 'khởi nghiệp'],
            
            # Technology
            'technology': ['trí tuệ nhân tạo', 'học máy', 'blockchain', 'điện toán đám mây', 'IoT', '5G'],
            
            # Benefits
            'benefit': ['cải thiện hiệu quả', 'tiết kiệm thời gian', 'tăng năng suất', 'giải quyết vấn đề'],
            
            # Fields
            'field': ['giáo dục', 'y tế', 'kinh doanh', 'nông nghiệp', 'du lịch', 'giao thông'],
            
            # Programming languages
            'programming_language': ['Python', 'JavaScript', 'Java', 'C++', 'React', 'Node.js'],
            
            # Learning difficulty
            'learning_difficulty': ['dễ học', 'khó', 'thách thức', 'phức tạp', 'đòi hỏi kiên nhẫn'],
            
            # Rewards
            'reward': ['thú vị', 'bổ ích', 'đáng giá', 'mang lại cơ hội tốt', 'phát triển tư duy'],
            
            # Topics
            'topic': ['công việc', 'cuộc sống', 'gia đình', 'sở thích', 'kế hoạch tương lai', 'kinh nghiệm'],
            
            # Reactions
            'reaction': ['thú vị', 'bổ ích', 'ấn tượng', 'học hỏi được nhiều', 'đáng suy ngẫm'],
            
            # Family activities
            'family_activity': ['ăn cơm cùng nhau', 'đi du lịch', 'xem phim', 'nấu ăn', 'dọn dẹp nhà cửa'],
            
            # Family time
            'family_time': ['cuối tuần', 'ngày nghỉ', 'buổi tối', 'dịp lễ', 'kỳ nghỉ hè'],
            
            # Social activities
            'social_activity': ['gặp gỡ', 'trò chuyện', 'ăn uống', 'chơi game', 'thể thao', 'karaoke'],
            
            # Social venues
            'social_venue': ['quán cà phê', 'nhà hàng', 'công viên', 'trung tâm thương mại', 'sân thể thao']
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