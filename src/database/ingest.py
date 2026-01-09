"""向量数据库构建模块 - 使用本地 GPU 进行 Embedding 计算。

本模块使用 BAAI/bge-m3 模型在本地 GPU (A4000) 上生成向量嵌入，
并将文本块存储到 ChromaDB 向量数据库中。

缓存目录配置（按优先级）:
    1. 环境变量 HF_HOME 或 TRANSFORMERS_CACHE（如果已设置）
    2. .venv/.cache/huggingface（如果 .venv 目录存在）
    3. ~/.cache/huggingface（默认位置）
    
    自动使用 .venv 可以避免用户主目录的磁盘配额问题。
    如果遇到磁盘配额问题，也可以手动设置环境变量：
    export HF_HOME=/path/to/larger/disk/.cache/huggingface
    export TRANSFORMERS_CACHE=/path/to/larger/disk/.cache/huggingface

模型信息:
    - 模型名称: BAAI/bge-m3
    - 模型大小: ~2.3GB
    - 向量维度: 1024
    - 支持语言: 多语言（包括德语和英语）
"""

import os
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 修复 SQLite 版本问题：使用 pysqlite3 替代系统 sqlite3
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # 如果 pysqlite3 不可用，使用系统 sqlite3

# 在导入 HuggingFace 相关库之前设置缓存和临时目录
# 这样可以确保所有下载和临时文件都存储在 .venv 中
def get_hf_cache_dir() -> str:
    """获取 HuggingFace 缓存目录。
    
    优先级：
    1. 环境变量 HF_HOME 或 TRANSFORMERS_CACHE
    2. .venv/.cache/huggingface（如果 .venv 存在）
    3. ~/.cache/huggingface（默认）
    
    Returns:
        缓存目录路径（绝对路径）
    """
    # 检查环境变量
    cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    
    if not cache_dir:
        # 检查是否存在 .venv 目录
        venv_cache = Path(".venv/.cache/huggingface")
        if Path(".venv").exists():
            cache_dir = str(venv_cache.absolute())
            venv_cache.mkdir(parents=True, exist_ok=True)
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    return cache_dir


def setup_hf_environment() -> None:
    """设置 HuggingFace 相关的环境变量，将缓存和临时文件都存储到 .venv 中。"""
    cache_dir = get_hf_cache_dir()
    
    # 设置 HuggingFace 缓存目录
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    # 设置临时目录（HuggingFace 下载时会使用）
    # 使用 .venv/.tmp 作为临时目录，避免使用用户主目录的临时目录
    if Path(".venv").exists():
        tmp_dir = str(Path(".venv/.tmp").absolute())
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = tmp_dir
        os.environ["TMP"] = tmp_dir
        os.environ["TEMP"] = tmp_dir

# 提前调用以设置环境变量
setup_hf_environment()

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger
import torch
from tqdm import tqdm


class KantVectorDB:
    """康德文本向量数据库类。
    
    使用本地 GPU 运行 BAAI/bge-m3 模型进行向量嵌入计算，
    并将结果存储到 ChromaDB 持久化数据库中。
    """

    def __init__(
        self,
        persist_directory: str = "./data/chromadb",
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 256,
    ) -> None:
        """初始化向量数据库。
        
        Args:
            persist_directory: ChromaDB 持久化存储目录
            model_name: HuggingFace 模型名称，默认为 BAAI/bge-m3
            batch_size: 批处理大小，默认为 256（针对 A4000 16GB 优化）
        
        Raises:
            RuntimeError: 当 GPU 不可用但尝试使用 CUDA 时
        """
        # 1. 检查 GPU 状态
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️  Hardware Check: Running on {device.upper()}")
        
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   GPU: {gpu_name}")
            logger.info(f"   VRAM: {vram_gb:.2f} GB")
        else:
            logger.warning("⚠️  No GPU detected, falling back to CPU (slower)")

        # 2. 初始化本地 Embedding 模型 (BGE-M3)
        # 环境变量已在模块导入时设置，这里只显示信息
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or get_hf_cache_dir()
        tmp_dir = os.getenv("TMPDIR") or os.getenv("TMP")
        
        # 显示使用的缓存和临时目录
        logger.info(f"📁 HuggingFace cache: {cache_dir}")
        if tmp_dir and Path(".venv").exists() and tmp_dir == str(Path(".venv/.tmp").absolute()):
            logger.info(f"📁 Temporary files: {tmp_dir} (using .venv to avoid quota issues)")
        elif tmp_dir:
            logger.info(f"📁 Temporary files: {tmp_dir}")
        
        logger.info(f"📥 Loading local embedding model ({model_name})...")
        logger.info("   Note: Model size ~2.3GB. First download may take time.")
        
        try:
            self.embedding_fn = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},  # 归一化有助于余弦相似度计算
            )
            logger.success(f"✅ Model loaded successfully on {device}")
        except RuntimeError as e:
            error_msg = str(e)
            if "Disk quota exceeded" in error_msg or "os error 122" in error_msg:
                logger.error("❌ Disk quota exceeded during model download!")
                logger.error("")
                logger.error("💡 Solutions:")
                logger.error("   1. Free up disk space (need ~3GB for model download)")
                logger.error("   2. Set custom cache directory with more space:")
                logger.error("      export HF_HOME=/path/to/larger/disk/.cache/huggingface")
                logger.error("      export TRANSFORMERS_CACHE=/path/to/larger/disk/.cache/huggingface")
                logger.error("   3. Or download model manually and place in cache directory")
                logger.error("")
                logger.error("   Model repository: https://huggingface.co/BAAI/bge-m3")
                raise RuntimeError(
                    "Disk quota exceeded. Please free up space or set HF_HOME/TRANSFORMERS_CACHE "
                    "to a directory with sufficient space."
                ) from e
            else:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            logger.error("")
            logger.error("💡 Troubleshooting:")
            logger.error("   - Check internet connection for model download")
            logger.error("   - Verify disk space availability (~3GB needed)")
            logger.error("   - Check HuggingFace access (may need login)")
            raise

        # 3. 初始化 ChromaDB
        # 注意：BGE-M3 的维度是 1024 (Dense)，不同于 OpenAI 的 1536/3072
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 Initializing ChromaDB at {persist_directory}...")
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        
        self.collection = self.client.get_or_create_collection(
            name="kant_corpus_local",
            metadata={"hnsw:space": "cosine", "model": model_name},
        )
        logger.info(f"✅ ChromaDB collection 'kant_corpus_local' ready")
        
        self.batch_size = batch_size

    def _flatten_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """将元数据扁平化，处理 list 类型。
        
        ChromaDB 的 metadata 字段不支持 list 类型，需要转换为字符串。
        
        Args:
            meta: 原始元数据字典
        
        Returns:
            扁平化后的元数据字典
        """
        clean_meta: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, list):
                clean_meta[k] = ", ".join(map(str, v))
            elif v is None:
                clean_meta[k] = ""
            else:
                clean_meta[k] = str(v)
        return clean_meta

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """批量写入文本块到向量数据库。
        
        使用 GPU 批处理进行向量嵌入计算，提高效率。
        
        Args:
            chunks: 文本块列表，每个元素包含 'id', 'content', 'metadata' 字段
        
        Raises:
            ValueError: 当 chunks 为空或格式不正确时
        """
        if not chunks:
            logger.warning("⚠️  No chunks to ingest")
            return

        logger.info(f"🚀 Starting LOCAL ingestion of {len(chunks)} chunks...")
        logger.info(f"   Batch size: {self.batch_size}")

        batch_ids: List[str] = []
        batch_documents: List[str] = []
        batch_metadatas: List[Dict[str, Any]] = []

        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        processed = 0

        with tqdm(total=len(chunks), desc="Embedding on GPU") as pbar:
            for chunk in chunks:
                # 验证 chunk 格式
                if not all(key in chunk for key in ["id", "content", "metadata"]):
                    logger.warning(f"⚠️  Skipping invalid chunk: missing required fields")
                    continue

                batch_ids.append(chunk["id"])
                batch_documents.append(chunk["content"])
                batch_metadatas.append(self._flatten_metadata(chunk["metadata"]))

                # 达到批处理大小时，执行嵌入和写入
                if len(batch_ids) >= self.batch_size:
                    self._upsert_batch(batch_ids, batch_documents, batch_metadatas)
                    processed += len(batch_ids)
                    pbar.update(len(batch_ids))
                    
                    batch_ids = []
                    batch_documents = []
                    batch_metadatas = []

            # 处理剩余的数据
            if batch_ids:
                self._upsert_batch(batch_ids, batch_documents, batch_metadatas)
                processed += len(batch_ids)
                pbar.update(len(batch_ids))

        final_count = self.collection.count()
        logger.success(
            f"✅ Ingestion complete. Processed {processed} chunks. "
            f"Collection count: {final_count}"
        )

    def _upsert_batch(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """执行单批数据的嵌入和写入。
        
        Args:
            ids: 文档 ID 列表
            documents: 文档内容列表
            metadatas: 元数据列表
        """
        try:
            # 使用本地模型计算向量
            embeddings = self.embedding_fn.embed_documents(documents)
            
            # 写入 ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error(f"❌ Failed to upsert batch: {e}")
            raise


def load_chunks_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件加载文本块。
    
    Args:
        file_path: JSONL 文件路径
    
    Returns:
        文本块列表
    
    Raises:
        FileNotFoundError: 当文件不存在时
        json.JSONDecodeError: 当 JSON 解析失败时
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    chunks: List[Dict[str, Any]] = []
    logger.info(f"📂 Loading chunks from {file_path}...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️  Failed to parse line {line_num} in {file_path}: {e}")
                    continue
        
        logger.info(f"✅ Loaded {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        logger.error(f"❌ Error reading {file_path}: {e}")
        raise


def find_all_chunk_files(chunks_dir: str = "data/chunks") -> List[str]:
    """查找所有 JSONL 格式的 chunk 文件。
    
    Args:
        chunks_dir: chunks 目录路径
    
    Returns:
        文件路径列表，按文件名排序
    """
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        logger.warning(f"⚠️  Chunks directory not found: {chunks_dir}")
        return []

    jsonl_files = sorted(chunks_path.glob("*.jsonl"))
    file_paths = [str(f) for f in jsonl_files]
    
    logger.info(f"📁 Found {len(file_paths)} JSONL files in {chunks_dir}")
    return file_paths


def check_disk_space(path: str, required_gb: float = 3.0) -> bool:
    """检查指定路径的磁盘空间是否足够。
    
    Args:
        path: 要检查的路径
        required_gb: 需要的空间（GB）
    
    Returns:
        如果空间足够返回 True，否则返回 False
    """
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        logger.info(f"💾 Disk space check for {path}:")
        logger.info(f"   Free: {free_gb:.2f} GB, Required: {required_gb:.2f} GB")
        return free_gb >= required_gb
    except Exception as e:
        logger.warning(f"⚠️  Could not check disk space: {e}")
        return True  # 假设空间足够，继续执行


def main() -> None:
    """主函数：执行向量数据库构建流程。"""
    # 配置
    DB_PATH = "./data/chromadb"
    CHUNKS_DIR = "data/chunks"
    
    # 确定缓存目录
    cache_dir = get_hf_cache_dir()
    
    logger.info("📋 Configuration:")
    logger.info(f"   ChromaDB path: {DB_PATH}")
    logger.info(f"   HuggingFace cache: {cache_dir}")
    if Path(".venv").exists() and cache_dir == str(Path(".venv/.cache/huggingface").absolute()):
        logger.info("   ✅ Using .venv for model storage (avoids quota issues)")
    
    # 检查磁盘空间
    if not check_disk_space(cache_dir, required_gb=3.0):
        logger.warning("⚠️  Insufficient disk space in cache directory!")
        logger.warning("   Set HF_HOME or TRANSFORMERS_CACHE to a directory with more space.")
        logger.warning("   Example: export HF_HOME=/path/to/larger/disk/.cache/huggingface")
    
    # 检查并清理旧数据库（如果存在且维度不匹配）
    if os.path.exists(DB_PATH):
        logger.warning(f"⚠️  Found existing DB at {DB_PATH}")
        logger.info("   Deleting old database (BGE-M3 uses 1024 dimensions, incompatible with OpenAI embeddings)")
        try:
            shutil.rmtree(DB_PATH)
            logger.info("🗑️  Deleted old database.")
        except Exception as e:
            logger.error(f"❌ Failed to delete old database: {e}")
            raise

    # 查找所有 chunk 文件
    chunk_files = find_all_chunk_files(CHUNKS_DIR)
    if not chunk_files:
        logger.error(f"❌ No chunk files found in {CHUNKS_DIR}")
        return

    # 加载所有 chunks
    all_chunks: List[Dict[str, Any]] = []
    for chunk_file in chunk_files:
        try:
            chunks = load_chunks_from_jsonl(chunk_file)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"❌ Failed to load {chunk_file}: {e}")
            continue

    if not all_chunks:
        logger.error("❌ No chunks loaded. Exiting.")
        return

    logger.info(f"📊 Total chunks to ingest: {len(all_chunks)}")

    # 初始化数据库并执行写入
    try:
        db = KantVectorDB(persist_directory=DB_PATH, batch_size=256)
        db.ingest_chunks(all_chunks)
        logger.success("🎉 Vector database construction completed successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to build vector database: {e}")
        raise


if __name__ == "__main__":
    main()
