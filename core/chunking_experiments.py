#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunking Strategy Experiments for RAG System

Compare different chunking strategies and evaluate their performance.
"""

import time
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd
from enhanced_text_processor import EnhancedTextProcessor, TextChunk

@dataclass
class ChunkingConfig:
    """Chunking configuration for experiments"""
    name: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_chunk_size: int
    enable_chinese_segmentation: bool = True
    enable_keyword_extraction: bool = True
    preserve_code_blocks: bool = True

@dataclass
class ExperimentResult:
    """Results from chunking experiment"""
    config_name: str
    total_chunks: int
    avg_chunk_size: int
    avg_token_count: float
    avg_importance_score: float
    processing_time: float
    chunks_in_optimal_range: int  # 20-500 tokens
    language_distribution: Dict[str, int]
    
class ChunkingExperiments:
    """Chunking strategy experiments and evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Predefined experimental configurations
        self.experiment_configs = [
            ChunkingConfig("Small_Chunks", 400, 50, 50, 600),
            ChunkingConfig("Medium_Chunks", 800, 100, 100, 1200),
            ChunkingConfig("Large_Chunks", 1200, 150, 150, 1800),
            ChunkingConfig("High_Overlap", 800, 200, 100, 1200),
            ChunkingConfig("Low_Overlap", 800, 50, 100, 1200),
            ChunkingConfig("No_Chinese_Seg", 800, 100, 100, 1200, False),
            ChunkingConfig("No_Keywords", 800, 100, 100, 1200, True, False),
        ]
    
    def run_experiment(self, test_data: List[Dict[str, Any]], 
                      configs: List[ChunkingConfig] = None) -> List[ExperimentResult]:
        """Run chunking experiments with different configurations"""
        
        if configs is None:
            configs = self.experiment_configs
        
        results = []
        
        self.logger.info(f" Starting chunking experiments with {len(configs)} configurations")
        
        for config in configs:
            self.logger.info(f" Testing configuration: {config.name}")
            
            # Create processor with current config
            processor_config = asdict(config)
            processor_config.pop('name')  # Remove name field
            processor = EnhancedTextProcessor(processor_config)
            
            # Measure processing time
            start_time = time.time()
            chunks = processor.process_search_results(test_data)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            result = self._calculate_metrics(config.name, chunks, processing_time)
            results.append(result)
            
            self.logger.info(f" {config.name}: {result.total_chunks} chunks, "
                           f"{result.avg_chunk_size} avg size, "
                           f"{result.processing_time:.2f}s")
        
        return results
    
    def _calculate_metrics(self, config_name: str, chunks: List[TextChunk], 
                          processing_time: float) -> ExperimentResult:
        """Calculate metrics for experiment result"""
        
        if not chunks:
            return ExperimentResult(
                config_name=config_name,
                total_chunks=0,
                avg_chunk_size=0,
                avg_token_count=0.0,
                avg_importance_score=0.0,
                processing_time=processing_time,
                chunks_in_optimal_range=0,
                language_distribution={}
            )
        
        # Basic metrics
        total_chunks = len(chunks)
        avg_chunk_size = sum(len(chunk.content) for chunk in chunks) // total_chunks
        avg_token_count = sum(chunk.token_count for chunk in chunks) / total_chunks
        avg_importance_score = sum(chunk.importance_score for chunk in chunks) / total_chunks
        
        # Quality metrics
        chunks_in_optimal_range = sum(1 for chunk in chunks 
                                    if 20 <= chunk.token_count <= 500)
        
        # Language distribution
        language_dist = {}
        for chunk in chunks:
            lang = chunk.language
            language_dist[lang] = language_dist.get(lang, 0) + 1
        
        return ExperimentResult(
            config_name=config_name,
            total_chunks=total_chunks,
            avg_chunk_size=avg_chunk_size,
            avg_token_count=avg_token_count,
            avg_importance_score=avg_importance_score,
            processing_time=processing_time,
            chunks_in_optimal_range=chunks_in_optimal_range,
            language_distribution=language_dist
        )
    
    def compare_results(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Compare experiment results in a DataFrame"""
        
        data = []
        for result in results:
            data.append({
                'Configuration': result.config_name,
                'Total Chunks': result.total_chunks,
                'Avg Chunk Size': result.avg_chunk_size,
                'Avg Token Count': f"{result.avg_token_count:.1f}",
                'Avg Importance Score': f"{result.avg_importance_score:.2f}",
                'Processing Time (s)': f"{result.processing_time:.2f}",
                'Optimal Range %': f"{(result.chunks_in_optimal_range/result.total_chunks*100):.1f}%" if result.total_chunks > 0 else "0%"
            })
        
        df = pd.DataFrame(data)
        return df
    
    def visualize_results(self, results: List[ExperimentResult], save_path: str = None):
        """Visualize experiment results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Chunking Strategy Comparison', fontsize=16)
        
        configs = [r.config_name for r in results]
        
        # 1. Total chunks comparison
        total_chunks = [r.total_chunks for r in results]
        axes[0, 0].bar(configs, total_chunks)
        axes[0, 0].set_title('Total Chunks Generated')
        axes[0, 0].set_ylabel('Number of Chunks')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Average token count
        avg_tokens = [r.avg_token_count for r in results]
        axes[0, 1].bar(configs, avg_tokens)
        axes[0, 1].set_title('Average Token Count')
        axes[0, 1].set_ylabel('Tokens')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Processing time
        proc_times = [r.processing_time for r in results]
        axes[1, 0].bar(configs, proc_times)
        axes[1, 0].set_title('Processing Time')
        axes[1, 0].set_ylabel('Seconds')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Quality score (optimal range percentage)
        quality_scores = [r.chunks_in_optimal_range/r.total_chunks*100 if r.total_chunks > 0 else 0 
                         for r in results]
        axes[1, 1].bar(configs, quality_scores)
        axes[1, 1].set_title('Chunks in Optimal Range (20-500 tokens)')
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f" Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: List[ExperimentResult], filepath: str):
        """Save experiment results to JSON file"""
        
        results_data = []
        for result in results:
            results_data.append(asdict(result))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> List[ExperimentResult]:
        """Load experiment results from JSON file"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        results = []
        for data in results_data:
            results.append(ExperimentResult(**data))
        
        self.logger.info(f"📂 Results loaded from {filepath}")
        return results

def create_test_dataset() -> List[Dict[str, Any]]:
    """Create a comprehensive test dataset for experiments"""
    
    return [
        {
            "title": "人工智能技术发展现状与趋势",
            "content": """
            人工智能（Artificial Intelligence, AI）作为21世纪最重要的技术革命之一，正在深刻改变着我们的生活和工作方式。
            
            ## 技术发展现状
            
            目前，AI技术主要包括以下几个核心领域：
            
            1. **机器学习（Machine Learning）**：通过算法让计算机从数据中学习模式
            2. **深度学习（Deep Learning）**：基于神经网络的学习方法
            3. **自然语言处理（NLP）**：让计算机理解和生成人类语言
            4. **计算机视觉（Computer Vision）**：让计算机"看懂"图像和视频
            
            ### 主要应用场景
            
            - 智能推荐系统：如电商平台的商品推荐
            - 语音助手：Siri、Alexa等智能语音交互系统
            - 自动驾驶：Tesla、Waymo等公司的无人驾驶技术
            - 医疗诊断：AI辅助医生进行疾病诊断和治疗方案制定
            
            ## 未来发展趋势
            
            1. **多模态AI**：结合文本、图像、语音等多种数据类型
            2. **边缘计算**：将AI计算能力部署到设备端
            3. **可解释AI**：让AI的决策过程更加透明和可理解
            4. **AI伦理**：确保AI技术的公平性和安全性
            
            The future of AI looks promising with continuous advancements in hardware and algorithms.
            """,
            "url": "https://example.com/ai-trends"
        },
        {
            "title": "Python Web开发最佳实践",
            "content": """
            Python作为一门优秀的编程语言，在Web开发领域有着广泛的应用。本文将介绍Python Web开发的最佳实践。
            
            ## 框架选择
            
            ### Django
            Django是一个高级的Python Web框架，遵循"约定优于配置"的原则：
            
            ```python
            from django.http import HttpResponse
            from django.shortcuts import render
            
            def index(request):
                return HttpResponse("Hello, world!")
            ```
            
            **优点**：
            - 功能完整，包含ORM、模板引擎、用户认证等
            - 安全性高，内置CSRF保护、SQL注入防护等
            - 社区活跃，文档完善
            
            ### Flask
            Flask是一个轻量级的Web框架：
            
            ```python
            from flask import Flask
            app = Flask(__name__)
            
            @app.route('/')
            def hello_world():
                return 'Hello, World!'
            ```
            
            **特点**：
            - 简单易学，适合小型项目
            - 灵活性高，可以自由选择组件
            - 扩展丰富，生态系统完善
            
            ## 开发规范
            
            1. **代码结构**：采用MVC或MVT模式
            2. **数据库设计**：合理设计表结构，使用索引优化查询
            3. **API设计**：遵循RESTful规范
            4. **测试**：编写单元测试和集成测试
            5. **部署**：使用Docker容器化部署
            
            ## 性能优化
            
            - 使用缓存（Redis、Memcached）
            - 数据库查询优化
            - 静态文件CDN加速
            - 异步处理（Celery）
            """,
            "url": "https://example.com/python-web-best-practices"
        },
        {
            "title": "区块链技术原理与应用",
            "content": """
            区块链（Blockchain）是一种分布式账本技术，具有去中心化、不可篡改、透明公开等特点。
            
            ## 核心概念
            
            ### 区块结构
            每个区块包含以下信息：
            - 区块头（Block Header）
            - 交易数据（Transaction Data）
            - 时间戳（Timestamp）
            - 前一个区块的哈希值（Previous Hash）
            
            ### 共识机制
            1. **工作量证明（PoW）**：比特币采用的共识机制
            2. **权益证明（PoS）**：以太坊2.0采用的机制
            3. **委托权益证明（DPoS）**：EOS等平台使用
            
            ## 技术特点
            
            - **去中心化**：没有中央控制机构
            - **不可篡改**：历史记录无法修改
            - **透明性**：所有交易公开可查
            - **安全性**：密码学保证数据安全
            
            ## 应用领域
            
            ### 金融服务
            - 数字货币：Bitcoin、Ethereum等
            - 跨境支付：降低转账成本和时间
            - 供应链金融：提高透明度和效率
            
            ### 其他应用
            - 供应链管理：商品溯源和防伪
            - 数字身份：身份认证和隐私保护
            - 智能合约：自动执行合约条款
            - 版权保护：数字作品确权
            
            Blockchain technology is revolutionizing various industries beyond cryptocurrency.
            """,
            "url": "https://example.com/blockchain-principles"
        }
    ]

def run_comprehensive_experiment():
    """Run comprehensive chunking experiments"""
    
    print(" Starting Comprehensive Chunking Experiments")
    print("=" * 60)
    
    # Initialize experiment framework
    experiments = ChunkingExperiments()
    
    # Create test dataset
    test_data = create_test_dataset()
    print(f" Test dataset: {len(test_data)} documents")
    
    # Run experiments
    results = experiments.run_experiment(test_data)
    
    # Display results
    print("\nExperiment Results:")
    print("=" * 60)
    
    df = experiments.compare_results(results)
    print(df.to_string(index=False))
    
    # Find best configuration
    best_config = max(results, key=lambda r: r.chunks_in_optimal_range/r.total_chunks if r.total_chunks > 0 else 0)
    print(f"\n🏆 Best Configuration: {best_config.config_name}")
    print(f"   - Optimal Range Coverage: {best_config.chunks_in_optimal_range/best_config.total_chunks*100:.1f}%")
    print(f"   - Average Importance Score: {best_config.avg_importance_score:.2f}")
    print(f"   - Processing Time: {best_config.processing_time:.2f}s")
    
    # Save results
    experiments.save_results(results, "chunking_experiment_results.json")
    
    # Visualize results (optional, requires matplotlib)
    try:
        experiments.visualize_results(results, "chunking_comparison.png")
    except ImportError:
        print(" Matplotlib not available, skipping visualization")
    
    return results

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run experiments
    results = run_comprehensive_experiment()