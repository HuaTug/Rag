
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Text Processor for RAG System

Provides intelligent text chunking, cleaning, and preprocessing for better vector embedding.
"""

import re
import logging
import jieba
import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

@dataclass
class TextChunk:
    """Text chunk with metadata"""
    content: str
    title: str = ""
    url: str = ""
    chunk_id: int = 0
    token_count: int = 0
    language: str = "mixed"
    importance_score: float = 1.0
    keywords: List[str] = None
    metadata: Dict[str, Any] = None

class EnhancedTextProcessor:
    """Enhanced text processor with intelligent chunking"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Chunking parameters
        self.chunk_size = self.config.get("chunk_size", 800)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.min_chunk_size = self.config.get("min_chunk_size", 100)
        self.max_chunk_size = self.config.get("max_chunk_size", 1200)
        
        # Language processing
        self.enable_chinese_segmentation = self.config.get("enable_chinese_segmentation", True)
        self.enable_keyword_extraction = self.config.get("enable_keyword_extraction", True)
        self.preserve_code_blocks = self.config.get("preserve_code_blocks", True)
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize text processing components"""
        try:
            # Initialize jieba for Chinese segmentation
            if self.enable_chinese_segmentation:
                jieba.initialize()
                self.logger.info(" Jieba Chinese segmentation initialized")
            
            # Initialize spaCy for English processing (optional)
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
                self.logger.info(" SpaCy English model loaded")
            except OSError:
                self.nlp_en = None
                self.logger.warning(" SpaCy English model not found, using basic processing")
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self._smart_length_function,
                separators=[
                    "\n\n\n",  # Multiple line breaks
                    "\n\n",    # Double line breaks
                    "\n",      # Single line breaks
                    "。",      # Chinese period
                    "！",      # Chinese exclamation
                    "？",      # Chinese question mark
                    ". ",      # English period with space
                    "! ",      # English exclamation with space
                    "? ",      # English question mark with space
                    "；",      # Chinese semicolon
                    "; ",      # English semicolon with space
                    "，",      # Chinese comma
                    ", ",      # English comma with space
                    " ",       # Space
                    ""         # Character level (fallback)
                ]
            )
            
        except Exception as e:
            self.logger.error(f" Failed to initialize components: {e}")
    
    def process_search_results(self, search_results: List[Dict[str, Any]]) -> List[TextChunk]:
        """Process search results into optimized text chunks"""
        all_chunks = []
        
        for i, result in enumerate(search_results):
            try:
                # Extract and clean content
                raw_content = result.get('content', '')
                title = result.get('title', '')
                url = result.get('url', '')
                
                # Clean and preprocess content
                cleaned_content = self._clean_content(raw_content)
                
                if len(cleaned_content.strip()) < self.min_chunk_size:
                    self.logger.debug(f"Skipping short content from {url}")
                    continue
                
                # Intelligent chunking
                chunks = self._intelligent_chunk(cleaned_content, title, url)
                
                # Add chunk metadata
                for j, chunk in enumerate(chunks):
                    chunk.chunk_id = len(all_chunks) + j
                    chunk.metadata = {
                        'source_index': i,
                        'total_chunks_from_source': len(chunks),
                        'chunk_index_in_source': j,
                        'original_length': len(raw_content),
                        'processed_length': len(cleaned_content)
                    }
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                self.logger.error(f" Error processing search result {i}: {e}")
                continue
        
        self.logger.info(f" Processed {len(search_results)} search results into {len(all_chunks)} chunks")
        return all_chunks
    
    def _clean_content(self, content: str) -> str:
        """Clean and preprocess content"""
        if not content:
            return ""
        
        # Remove HTML tags if present
        if '<' in content and '>' in content:
            content = self._clean_html(content)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove excessive punctuation
        content = re.sub(r'[.]{3,}', '...', content)
        content = re.sub(r'[!]{2,}', '!', content)
        content = re.sub(r'[?]{2,}', '?', content)
        
        # Clean up common web artifacts
        content = re.sub(r'(Cookie|cookie)s?\s+(policy|notice|consent)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'(Privacy|privacy)\s+(policy|notice)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'(Terms|terms)\s+(of\s+)?(service|use)', '', content, flags=re.IGNORECASE)
        
        # Remove navigation elements
        nav_patterns = [
            r'(Home|首页)\s*>\s*',
            r'(Back to top|返回顶部)',
            r'(Share|分享)\s*:',
            r'(Follow us|关注我们)',
            r'(Subscribe|订阅)',
        ]
        for pattern in nav_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract meaningful text"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'advertisement', 'ads', 'menu']
            for tag_name in unwanted_tags:
                for tag in soup.find_all(tag_name):
                    tag.decompose()
            
            # Remove elements by class/id patterns
            unwanted_patterns = ['nav', 'menu', 'sidebar', 'footer', 'header', 
                               'ad', 'advertisement', 'cookie', 'popup']
            for pattern in unwanted_patterns:
                for tag in soup.find_all(attrs={'class': re.compile(pattern, re.I)}):
                    tag.decompose()
                for tag in soup.find_all(attrs={'id': re.compile(pattern, re.I)}):
                    tag.decompose()
            
            # Preserve code blocks if enabled
            if self.preserve_code_blocks:
                for code_tag in soup.find_all(['code', 'pre']):
                    code_tag.string = f"\n[CODE_BLOCK]\n{code_tag.get_text()}\n[/CODE_BLOCK]\n"
            
            # Extract main content
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '.content', 
                           '#content', '.main', '.post-content', '.entry-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Get text with some structure preservation
            text = main_content.get_text(separator=' ', strip=True)
            
            return text
            
        except Exception as e:
            self.logger.debug(f"HTML cleaning failed: {e}")
            # Fallback: simple tag removal
            return re.sub(r'<[^>]+>', '', html_content)
    
    def _intelligent_chunk(self, content: str, title: str = "", url: str = "") -> List[TextChunk]:
        """Intelligent chunking with semantic awareness"""
        chunks = []
        
        # Detect language mix
        language = self._detect_language(content)
        
        # Pre-process for better chunking
        if language in ['chinese', 'mixed'] and self.enable_chinese_segmentation:
            content = self._preprocess_chinese_text(content)
        
        # Use recursive character splitter with smart separators
        text_chunks = self.text_splitter.split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            # Create chunk object
            chunk = TextChunk(
                content=chunk_text.strip(),
                title=title,
                url=url,
                chunk_id=i,
                token_count=self._count_tokens(chunk_text),
                language=language,
                importance_score=self._calculate_importance_score(chunk_text, title),
                keywords=self._extract_keywords(chunk_text) if self.enable_keyword_extraction else [],
                metadata={}
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _preprocess_chinese_text(self, text: str) -> str:
        """Preprocess Chinese text for better chunking"""
        # Add spaces around Chinese punctuation for better splitting
        text = re.sub(r'([。！？；])', r'\1 ', text)
        
        # Segment Chinese text with jieba
        if self.enable_chinese_segmentation:
            # Only segment Chinese parts, preserve English words
            def segment_chinese_part(match):
                chinese_text = match.group(0)
                segmented = jieba.cut(chinese_text, cut_all=False)
                return ' '.join(segmented)
            
            # Find Chinese character sequences
            text = re.sub(r'[\u4e00-\u9fff]+', segment_chinese_part, text)
        
        return text
    
    def _detect_language(self, text: str) -> str:
        """Detect text language"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = chinese_chars + english_chars
        
        if total_chars == 0:
            return "unknown"
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.7:
            return "chinese"
        elif chinese_ratio < 0.3:
            return "english"
        else:
            return "mixed"
    
    def _smart_length_function(self, text: str) -> int:
        """Smart length function considering Chinese and English characters"""
        # Chinese characters count as 2, English as 1
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z0-9]', text))
        other_chars = len(text) - chinese_chars - english_chars
        
        return chinese_chars * 2 + english_chars + other_chars * 0.5
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Simple estimation: Chinese chars * 1.5 + English words
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        return int(chinese_chars * 1.5 + english_words)
    
    def _calculate_importance_score(self, text: str, title: str = "") -> float:
        """Calculate importance score for the chunk"""
        score = 1.0
        
        # Boost score if chunk contains title keywords
        if title:
            title_words = set(re.findall(r'\b\w+\b', title.lower()))
            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            overlap = len(title_words.intersection(text_words))
            if overlap > 0:
                score += 0.3 * (overlap / len(title_words))
        
        # Boost score for chunks with structured content
        if re.search(r'^\d+\.|\*|\-', text.strip(), re.MULTILINE):
            score += 0.2
        
        # Boost score for chunks with technical terms
        technical_patterns = [
            r'\b(API|SDK|HTTP|JSON|XML|SQL|AI|ML|DL)\b',
            r'\b\d+%\b',  # Percentages
            r'\$\d+',     # Prices
            r'\b\d{4}年\b'  # Years in Chinese
        ]
        for pattern in technical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
                break
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        keywords = []
        
        try:
            # Extract Chinese keywords with jieba
            if self.enable_chinese_segmentation:
                chinese_words = jieba.analyse.extract_tags(text, topK=5, withWeight=False)
                keywords.extend(chinese_words)
            
            # Extract English keywords
            english_words = re.findall(r'\b[A-Za-z]{3,}\b', text)
            word_freq = {}
            for word in english_words:
                word_lower = word.lower()
                if word_lower not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']:
                    word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
            
            # Get top English keywords
            top_english = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            keywords.extend([word for word, freq in top_english])
            
        except Exception as e:
            self.logger.debug(f"Keyword extraction failed: {e}")
        
        return keywords[:8]  # Limit to 8 keywords
    
    def optimize_for_embedding(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Optimize chunks for better embedding quality"""
        optimized_chunks = []
        
        for chunk in chunks:
            # Skip very short or very long chunks
            if chunk.token_count < 20 or chunk.token_count > 500:
                continue
            
            # Enhance chunk content with context
            enhanced_content = self._enhance_chunk_content(chunk)
            chunk.content = enhanced_content
            
            optimized_chunks.append(chunk)
        
        # Sort by importance score
        optimized_chunks.sort(key=lambda x: x.importance_score, reverse=True)
        
        self.logger.info(f" Optimized {len(chunks)} chunks to {len(optimized_chunks)} high-quality chunks")
        return optimized_chunks
    
    def _enhance_chunk_content(self, chunk: TextChunk) -> str:
        """Enhance chunk content with context information"""
        enhanced_content = chunk.content
        
        # Add title context if available and relevant
        if chunk.title and chunk.title.lower() not in enhanced_content.lower():
            # Check if title provides useful context
            title_words = set(re.findall(r'\b\w+\b', chunk.title.lower()))
            content_words = set(re.findall(r'\b\w+\b', enhanced_content.lower()))
            
            if len(title_words.intersection(content_words)) / len(title_words) < 0.5:
                enhanced_content = f"关于{chunk.title}: {enhanced_content}"
        
        # Add keyword context
        if chunk.keywords:
            key_terms = ', '.join(chunk.keywords[:3])
            if not any(keyword.lower() in enhanced_content.lower() for keyword in chunk.keywords[:3]):
                enhanced_content = f"[关键词: {key_terms}] {enhanced_content}"
        
        return enhanced_content

# Factory function
def create_enhanced_text_processor(config: Dict[str, Any] = None) -> EnhancedTextProcessor:
    """Create enhanced text processor with configuration"""
    return EnhancedTextProcessor(config)

# Test function
def test_text_processor():
    """Test the enhanced text processor"""
    print(" Testing Enhanced Text Processor")
    
    processor = create_enhanced_text_processor({
        "chunk_size": 600,
        "chunk_overlap": 100,
        "enable_chinese_segmentation": True,
        "enable_keyword_extraction": True
    })
    
    # Test data
    test_results = [
        {
            "title": "人工智能发展趋势",
            "content": "人工智能（AI）技术正在快速发展。机器学习和深度学习算法不断改进，使得AI系统能够处理更复杂的任务。Natural Language Processing (NLP) has made significant progress in recent years. 语言模型如GPT和BERT已经在各种应用中展现出强大的能力。",
            "url": "https://example.com/ai-trends"
        },
        {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data. 监督学习需要标记数据来训练模型，而无监督学习则在未标记的数据中寻找模式。",
            "url": "https://example.com/ml-basics"
        }
    ]
    
    # Process results
    chunks = processor.process_search_results(test_results)
    
    print(f" Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Title: {chunk.title}")
        print(f"Language: {chunk.language}")
        print(f"Token Count: {chunk.token_count}")
        print(f"Importance Score: {chunk.importance_score:.2f}")
        print(f"Keywords: {chunk.keywords}")
        print(f"Content: {chunk.content[:200]}...")
    
    # Test optimization
    optimized_chunks = processor.optimize_for_embedding(chunks)
    print(f"\n Optimized to {len(optimized_chunks)} high-quality chunks")

if __name__ == "__main__":
    test_text_processor()