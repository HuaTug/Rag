import sys
import os
import ssl
import certifi
from glob import glob
from tqdm import tqdm

from encoder import emb_text
from milvus_utils import get_milvus_client, create_collection

from dotenv import load_dotenv


load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT", "./milvus_demo.db")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")


def get_text(data_dir):
    """Load documents and split each into chunks.

    Return:
        A dictionary of text chunks with the filepath as key value.
    """
    text_dict = {}
    for file_path in glob(os.path.join(data_dir, "**/*.md"), recursive=True):
        if file_path.endswith(".md"):
            with open(file_path, "r") as file:
                file_text = file.read().strip()
            
            # 改进的文档切分策略
            chunks = []
            sections = file_text.split("# ")
            
            for i, section in enumerate(sections):
                if section.strip():  # 跳过空段落
                    if i > 0:  # 第一个section可能没有"# "前缀
                        section = "# " + section
                    
                    # 如果段落太短，尝试与下一个段落合并
                    if len(section.strip()) < 50 and i < len(sections) - 1:
                        next_section = sections[i + 1].strip()
                        if next_section:
                            section = section + "\n\n# " + next_section
                            sections[i + 1] = ""  # 标记已处理
                    
                    chunks.append(section.strip())
            
            text_dict[file_path] = [chunk for chunk in chunks if chunk.strip()]
    return text_dict


# Get Milvus client
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

# 设置使用开源嵌入模型
os.environ["USE_OPENSOURCE_EMBEDDING"] = "true"

print("✅ 使用开源嵌入模型 (sentence-transformers)")
print("📝 正在初始化嵌入模型...")

# Set SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Get text data from data directory
data_dir = sys.argv[-1]
text_dict = get_text(data_dir)

# Create collection - 测试嵌入维度
print("📏 检测嵌入向量维度...")
test_embedding = emb_text("test")
dim = len(test_embedding)
print(f"✅ 嵌入向量维度: {dim}")

create_collection(milvus_client=milvus_client, collection_name=COLLECTION_NAME, dim=dim)

# Insert data
data = []
count = 0
for i, filepath in enumerate(tqdm(text_dict, desc="Creating embeddings")):
    chunks = text_dict[filepath]
    for line in chunks:
        try:
            vector = emb_text(line)
            data.append({"vector": vector, "text": line})
            count += 1
        except Exception as e:
            print(
                f"Skipping file: {filepath} due to an error occurs during the embedding process:\n{e}"
            )
            continue
print("Total number of loaded documents:", count)

# Insert data into Milvus collection
mr = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
print("Total number of entities/chunks inserted:", mr["insert_count"])
