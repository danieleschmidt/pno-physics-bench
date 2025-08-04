"""Memory-efficient data loading and parallel processing."""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import h5py
from typing import Iterator, Tuple, Optional, List, Dict, Any, Union
import threading
import queue
import time
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

from .exceptions import DataError, ResourceError
from .logging_config import PerformanceLogger


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger()


class StreamingDataset(IterableDataset):
    """Memory-efficient streaming dataset for large PDE datasets."""
    
    def __init__(
        self,
        data_source: Union[str, List[str]],
        chunk_size: int = 1000,
        buffer_size: int = 5,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        preload_workers: int = 2,
    ):
        """Initialize streaming dataset.
        
        Args:
            data_source: Path to HDF5 file or list of paths
            chunk_size: Number of samples to load at once
            buffer_size: Number of chunks to buffer in memory
            transform: Optional transform for inputs
            target_transform: Optional transform for targets
            preload_workers: Number of background loading workers
        """
        super().__init__()
        
        if isinstance(data_source, str):
            self.data_files = [Path(data_source)]
        else:
            self.data_files = [Path(f) for f in data_source]
        
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.transform = transform
        self.target_transform = target_transform
        self.preload_workers = preload_workers
        
        # Validate data files
        for file_path in self.data_files:
            if not file_path.exists():
                raise DataError(f"Data file not found: {file_path}")
        
        # Get dataset info
        self.total_samples = 0
        self.input_shape = None
        self.output_shape = None
        self._get_dataset_info()
        
        logger.info(f"StreamingDataset initialized: {self.total_samples} samples across {len(self.data_files)} files")
    
    def _get_dataset_info(self) -> None:
        """Get dataset information from first file."""
        with h5py.File(self.data_files[0], 'r') as f:
            # Assume standard structure with 'inputs' and 'outputs' groups
            for split in ['train', 'val', 'test']:
                if split in f:
                    inputs = f[split]['inputs']
                    outputs = f[split]['outputs']
                    
                    self.total_samples += len(inputs)
                    
                    if self.input_shape is None:
                        self.input_shape = inputs[0].shape
                        self.output_shape = outputs[0].shape
                    
                    break
        
        # Count samples in remaining files
        for file_path in self.data_files[1:]:
            with h5py.File(file_path, 'r') as f:
                for split in ['train', 'val', 'test']:
                    if split in f:
                        self.total_samples += len(f[split]['inputs'])
                        break
    
    def _load_chunk(self, file_path: Path, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a chunk of data from file.
        
        Args:
            file_path: Path to data file
            start_idx: Start index
            end_idx: End index
            
        Returns:
            Tuple of (inputs, outputs)
        """
        with h5py.File(file_path, 'r') as f:
            # Find the appropriate split
            for split in ['train', 'val', 'test']:
                if split in f:
                    inputs = f[split]['inputs'][start_idx:end_idx]
                    outputs = f[split]['outputs'][start_idx:end_idx]
                    return inputs, outputs
        
        raise DataError(f"No valid split found in {file_path}")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over dataset samples."""
        # Create chunk buffer
        chunk_buffer = queue.Queue(maxsize=self.buffer_size)
        
        # Background thread for loading chunks
        def load_chunks():
            try:
                for file_path in self.data_files:
                    with h5py.File(file_path, 'r') as f:
                        for split in ['train', 'val', 'test']:
                            if split in f:
                                dataset_size = len(f[split]['inputs'])
                                
                                for start_idx in range(0, dataset_size, self.chunk_size):
                                    end_idx = min(start_idx + self.chunk_size, dataset_size)
                                    
                                    # Load chunk
                                    with perf_logger.timer("load_chunk", chunk_size=end_idx-start_idx):
                                        inputs, outputs = self._load_chunk(file_path, start_idx, end_idx)
                                    
                                    # Convert to tensors
                                    inputs = torch.from_numpy(inputs).float()
                                    outputs = torch.from_numpy(outputs).float()
                                    
                                    # Put in buffer (blocks if full)
                                    chunk_buffer.put((inputs, outputs))
                                
                                break  # Only process first valid split per file
                
                # Signal end of data
                chunk_buffer.put(None)
                
            except Exception as e:
                logger.error(f"Error loading chunks: {e}")
                chunk_buffer.put(None)
        
        # Start background loading
        loader_thread = threading.Thread(target=load_chunks, daemon=True)
        loader_thread.start()
        
        # Yield samples from buffered chunks
        try:
            while True:
                chunk = chunk_buffer.get()
                if chunk is None:
                    break
                
                inputs_chunk, outputs_chunk = chunk
                
                # Yield individual samples
                for i in range(len(inputs_chunk)):
                    input_sample = inputs_chunk[i]
                    output_sample = outputs_chunk[i]
                    
                    # Apply transforms
                    if self.transform:
                        input_sample = self.transform(input_sample)
                    if self.target_transform:
                        output_sample = self.target_transform(output_sample)
                    
                    yield input_sample, output_sample
        
        finally:
            # Clean up
            loader_thread.join(timeout=1.0)


class ParallelDataProcessor:
    """Parallel processing for data operations."""
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_processes: bool = False,
        chunk_size: int = 100,
    ):
        """Initialize parallel data processor.
        
        Args:
            num_workers: Number of workers (auto-detect if None)
            use_processes: Use processes instead of threads
            chunk_size: Size of processing chunks
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)
        
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        logger.info(f"ParallelDataProcessor initialized: {num_workers} {'processes' if use_processes else 'threads'}")
    
    def process_batch(
        self,
        data: List[Any],
        process_fn: callable,
        **kwargs
    ) -> List[Any]:
        """Process data batch in parallel.
        
        Args:
            data: List of data items to process
            process_fn: Processing function
            **kwargs: Additional arguments for process_fn
            
        Returns:
            List of processed results
        """
        if len(data) <= self.chunk_size:
            # Process directly if small enough
            if kwargs:
                process_fn = partial(process_fn, **kwargs)
            return list(self.executor.map(process_fn, data))
        
        # Split into chunks
        chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        
        # Process chunks in parallel
        def process_chunk(chunk):
            if kwargs:
                fn = partial(process_fn, **kwargs)
                return [fn(item) for item in chunk]
            else:
                return [process_fn(item) for item in chunk]
        
        chunk_results = list(self.executor.map(process_chunk, chunks))
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)


class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for efficient access to large files."""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_size: int = 1000,
    ):
        """Initialize memory-mapped dataset.
        
        Args:
            data_path: Path to HDF5 data file
            split: Dataset split to use
            transform: Optional transform for inputs
            target_transform: Optional transform for targets
            cache_size: Number of samples to cache in memory
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if not self.data_path.exists():
            raise DataError(f"Data file not found: {self.data_path}")
        
        # Open file and get dataset info
        self.file = h5py.File(self.data_path, 'r')
        
        if split not in self.file:
            raise DataError(f"Split '{split}' not found in {self.data_path}")
        
        self.inputs = self.file[split]['inputs']
        self.outputs = self.file[split]['outputs']
        self.length = len(self.inputs)
        
        # Simple LRU cache for recently accessed samples
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size
        
        logger.info(f"MemoryMappedDataset initialized: {self.length} samples from {split} split")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.length}")
        
        # Check cache first
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            input_data, output_data = self.cache[idx]
        else:
            # Load from disk
            with perf_logger.timer("disk_load", sample_idx=idx):
                input_data = torch.from_numpy(self.inputs[idx]).float()
                output_data = torch.from_numpy(self.outputs[idx]).float()
            
            # Add to cache
            if len(self.cache) >= self.cache_size:
                # Remove least recently used
                lru_idx = self.cache_order.pop(0)
                del self.cache[lru_idx]
            
            self.cache[idx] = (input_data.clone(), output_data.clone())
            self.cache_order.append(idx)
        
        # Apply transforms
        if self.transform:
            input_data = self.transform(input_data)
        if self.target_transform:
            output_data = self.target_transform(output_data)
        
        return input_data, output_data
    
    def __del__(self):
        """Clean up file handle."""
        if hasattr(self, 'file') and self.file:
            self.file.close()


class AdaptiveDataLoader:
    """Adaptive data loader that adjusts batch size based on memory usage."""
    
    def __init__(
        self,
        dataset: Dataset,
        initial_batch_size: int = 32,
        max_batch_size: int = 256,
        min_batch_size: int = 1,
        memory_threshold: float = 0.8,  # 80% of available GPU memory
        adaptation_factor: float = 0.9,
        **dataloader_kwargs
    ):
        """Initialize adaptive data loader.
        
        Args:
            dataset: Dataset to load from
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum allowed batch size
            memory_threshold: Memory usage threshold for adaptation
            adaptation_factor: Factor to reduce batch size when OOM
            **dataloader_kwargs: Additional DataLoader arguments
        """
        self.dataset = dataset
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.memory_threshold = memory_threshold
        self.adaptation_factor = adaptation_factor
        self.dataloader_kwargs = dataloader_kwargs
        
        # Track performance
        self.batch_times = []
        self.memory_usage = []
        
        logger.info(f"AdaptiveDataLoader initialized: batch_size={initial_batch_size}")
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction of total."""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return allocated / total
    
    def _create_dataloader(self, batch_size: int) -> DataLoader:
        """Create DataLoader with specified batch size."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            **self.dataloader_kwargs
        )
    
    def __iter__(self):
        """Iterate with adaptive batch sizing."""
        dataloader = self._create_dataloader(self.current_batch_size)
        iterator = iter(dataloader)
        
        while True:
            try:
                start_time = time.time()
                
                # Get next batch
                batch = next(iterator)
                
                # Track timing
                batch_time = time.time() - start_time
                self.batch_times.append(batch_time)
                
                # Track memory usage
                memory_usage = self._get_memory_usage()
                self.memory_usage.append(memory_usage)
                
                # Adapt batch size if needed
                if len(self.memory_usage) > 10:  # Wait for some history
                    avg_memory = np.mean(self.memory_usage[-10:])
                    
                    if avg_memory > self.memory_threshold and self.current_batch_size > self.min_batch_size:
                        # Reduce batch size
                        new_batch_size = max(
                            self.min_batch_size,
                            int(self.current_batch_size * self.adaptation_factor)
                        )
                        
                        if new_batch_size != self.current_batch_size:
                            logger.info(f"Reducing batch size: {self.current_batch_size} -> {new_batch_size}")
                            self.current_batch_size = new_batch_size
                            
                            # Create new dataloader
                            dataloader = self._create_dataloader(self.current_batch_size)
                            iterator = iter(dataloader)
                            continue
                    
                    elif avg_memory < self.memory_threshold * 0.7 and self.current_batch_size < self.max_batch_size:
                        # Increase batch size
                        new_batch_size = min(
                            self.max_batch_size,
                            int(self.current_batch_size * 1.2)
                        )
                        
                        if new_batch_size != self.current_batch_size:
                            logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
                            self.current_batch_size = new_batch_size
                            
                            # Create new dataloader
                            dataloader = self._create_dataloader(self.current_batch_size)
                            iterator = iter(dataloader)
                            continue
                
                yield batch
                
            except StopIteration:
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size and retry
                    torch.cuda.empty_cache()
                    
                    new_batch_size = max(
                        self.min_batch_size,
                        int(self.current_batch_size * self.adaptation_factor)
                    )
                    
                    logger.warning(f"OOM detected, reducing batch size: {self.current_batch_size} -> {new_batch_size}")
                    self.current_batch_size = new_batch_size
                    
                    # Create new dataloader
                    dataloader = self._create_dataloader(self.current_batch_size)
                    iterator = iter(dataloader)
                    continue
                else:
                    raise e
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance stats
        """
        stats = {
            'current_batch_size': self.current_batch_size,
        }
        
        if self.batch_times:
            stats.update({
                'avg_batch_time': np.mean(self.batch_times),
                'samples_per_second': self.current_batch_size / np.mean(self.batch_times),
            })
        
        if self.memory_usage:
            stats.update({
                'avg_memory_usage': np.mean(self.memory_usage),
                'max_memory_usage': np.max(self.memory_usage),
            })
        
        return stats


class PrefetchDataLoader:
    """Data loader with background prefetching for better GPU utilization."""
    
    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        prefetch_batches: int = 2,
    ):
        """Initialize prefetch data loader.
        
        Args:
            dataloader: Base DataLoader
            device: Target device for prefetching
            prefetch_batches: Number of batches to prefetch
        """
        self.dataloader = dataloader
        self.device = device
        self.prefetch_batches = prefetch_batches
        
    def __iter__(self):
        """Iterate with background prefetching."""
        # Create prefetch queue
        prefetch_queue = queue.Queue(maxsize=self.prefetch_batches)
        
        def prefetch_worker():
            """Background worker for prefetching data."""
            try:
                for batch in self.dataloader:
                    # Move to device
                    if isinstance(batch, (tuple, list)):
                        batch = tuple(item.to(self.device, non_blocking=True) for item in batch)
                    else:
                        batch = batch.to(self.device, non_blocking=True)
                    
                    prefetch_queue.put(batch)
                
                # Signal end
                prefetch_queue.put(None)
                
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                prefetch_queue.put(None)
        
        # Start prefetch worker
        worker_thread = threading.Thread(target=prefetch_worker, daemon=True)
        worker_thread.start()
        
        # Yield prefetched batches
        try:
            while True:
                batch = prefetch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            worker_thread.join(timeout=1.0)
    
    def __len__(self):
        return len(self.dataloader)