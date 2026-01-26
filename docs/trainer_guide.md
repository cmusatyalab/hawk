# Creating a New Model Trainer

This guide explains how to implement a new model trainer for Hawk using the setuptools entrypoint mechanism.

## Overview

Hawk uses a pluggable trainer architecture that allows different machine learning models to be used without modifying the core codebase. Trainers are discovered and loaded via setuptools entry points.

## Entry Point Registration

To register your trainer, add an entry point in `pyproject.toml` under the `[project.entry-points."cmuhawk.models"]` section:

```toml
[project.entry-points."cmuhawk.models"]
your_trainer_name = "your_trainer.trainer:YourTrainer"
```

Where:
- `your_trainer_name`: The name users will specify in their configuration (e.g., `train_strategy.type`)
- `your_trainer.trainer`: The full Python path to your trainer class
- `YourTrainer`: The name of your trainer class

## Required Files

Create a directory structure like:

```
your_trainer/
├── __init__.py
├── config.py
├── trainer.py
└── model.py
```

### 1. `__init__.py`

Create an empty `__init__.py` file:

```python
"""Your trainer package."""
```

### 2. `config.py`

Define configuration classes inheriting from Hawk's base configs. You typically need two classes:

```python
from pathlib import Path
from typing import Optional

from hawk.scout.core.config import ModelConfig, ModelTrainerConfig


class YourModelConfig(ModelConfig):
    """Model-specific configuration.

    Inherits from ModelConfig which provides:
    - mode: ModelMode (HAWK, ORACLE, NOTIONAL)
    """
    # Add model-specific parameters here
    param1: str = "default_value"
    param2: int = 10


class YourTrainerConfig(YourModelConfig, ModelTrainerConfig):
    """Trainer configuration.

    Inherits from both YourModelConfig and ModelTrainerConfig.
    ModelTrainerConfig provides:
    - initial_model_epochs: int
    - online_epochs: int | list[tuple[int, int]]
    - capture_trainingset: bool
    - capture_trainingset_compresslevel: int
    - notional_model_path: Optional[Path]
    - notional_train_time: float
    """
    # Add trainer-specific parameters here
    train_batch_size: int = 32
    learning_rate: float = 0.001
```

**Important**: The trainer config class must be set as `config_class` in your trainer class (see below).

### 3. `trainer.py`

Implement the trainer class inheriting from `ModelTrainerBase`:

```python
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from hawk.scout.core.model_trainer import ModelTrainer
from logzero import logger

from .config import YourTrainerConfig
from .model import YourModel

if TYPE_CHECKING:
    from hawk.scout.context.model_trainer_context import ModelContext


class YourTrainer(ModelTrainer):
    """Your trainer implementation.

    Inherits from ModelTrainer which provides:
    - Version management (get_new_version, get_version)
    - Model import/import_model methods
    - Model training state management
    - Training set capture functionality
    """

    config_class = YourTrainerConfig
    config: YourTrainerConfig

    def __init__(self, config: YourTrainerConfig, context: ModelContext) -> None:
        super().__init__(config, context)

        logger.info(f"Model_dir {self.context.model_dir}")
        logger.info("YOUR TRAINER CALLED")

    def load_model(self, path: Path, version: int) -> YourModel:
        """Load a trained model from disk.

        Args:
            path: Path to the model file
            version: Model version number

        Returns:
            YourModel instance ready for inference
        """
        logger.info(f"Loading from path {path}")
        return YourModel(self.config, self.context, path, version)

    def train_model(self, train_dir: Path) -> YourModel:
        """Train a new model.

        Args:
            train_dir: Directory containing training data

        Returns:
            Trained YourModel instance
        """
        new_version = self.get_new_version()
        model_savepath = self.context.model_path(new_version)

        # Access training data from context
        num_classes = len(self.context.class_list)
        labels = [str(label) for label in range(num_classes)]

        # Create training text file
        trainpath = self.context.model_path(new_version, template="train-{}.txt")
        train_len = self.make_train_txt(trainpath, train_dir, labels)

        # Prepare training command
        cmd = [
            sys.executable,
            "-m",
            "your_trainer.train",
            "--trainpath",
            str(trainpath),
            "--savepath",
            str(model_savepath),
            "--num-classes",
            str(num_classes),
            # Add any additional arguments
            "--batch-size",
            str(self.config.train_batch_size),
        ]

        # Capture training set for post-mission analysis
        capture_files = [trainpath, train_dir]
        self.capture_trainingset(shlex.join(cmd), capture_files)

        # Run training
        logger.info(f"TRAIN CMD\n{shlex.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Create model instance
        self.prev_model_path = model_savepath
        return YourModel(
            self.config,
            self.context,
            model_savepath,
            new_version,
            train_examples=train_len,
        )
```

**Key methods to implement**:

- `load_model(path, version)`: Load a saved model for inference
- `train_model(train_dir)`: Train a new model from data

**Inherited methods available**:

- `get_new_version()`: Get next model version number
- `get_version()`: Get current model version
- `import_model(bytes)`: Import model from bytes (used for remote models)
- `model_trainer(train_dir)`: Main training entry point
- `make_train_txt()`: Create training data file
- `get_num_epochs(version, positives)`: Get number of training epochs
- `capture_trainingset(cmd, extra_files)`: Archive training data

### 4. `model.py`

Implement the model class inheriting from `ModelBase`:

```python
import io
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import torch
from logzero import logger
from torch.utils.data import DataLoader

from hawk.detection import Detection
from hawk.scout.core.model import ModelBase
from hawk.scout.core.result_provider import ResultProvider
from hawk.scout.core.utils import log_exceptions

if TYPE_CHECKING:
    from hawk.hawkobject import HawkObject
    from hawk.objectid import ObjectId
    from hawk.scout.context.model_trainer_context import ModelContext
    from .config import YourModelConfig


class YourModel(ModelBase):
    """Your model implementation.

    Inherits from ModelBase which provides:
    - Inference queue management
    - Result collection
    - Request counting
    - Model lifecycle management
    """

    config: YourModelConfig

    def __init__(
        self,
        config: YourModelConfig,
        context: ModelContext,
        model_path: Path,
        version: int,
        *,
        train_examples: dict[str, int] | None = None,
        train_time: float = 0.0,
    ) -> None:
        logger.info(f"Loading Your Model from {model_path}")
        assert model_path.is_file()

        super().__init__(
            config,
            context,
            model_path,
            version,
            train_examples,
            train_time,
        )

        # Load your model here
        self._model = self.load_model(model_path)
        self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()
        self._running = True

    def load_model(self, model_path: Path) -> torch.nn.Module:
        """Load model weights and preprocessing.

        Args:
            model_path: Path to model file

        Returns:
            PyTorch model ready for inference
        """
        # Load your model architecture and weights
        model = torch.load(model_path)
        return model

    def preprocess(self, obj: HawkObject) -> torch.Tensor:
        """Preprocess input object for inference.

        Args:
            obj: HawkObject containing the input data

        Returns:
            Preprocessed tensor ready for inference
        """
        # Implement preprocessing based on your model
        # For images:
        import PIL.Image
        image = PIL.Image.open(io.BytesIO(obj.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self._preprocess(image)
        return tensor

    def serialize(self) -> bytes:
        """Serialize model for transmission to scouts.

        Returns:
            Serialized model bytes
        """
        if self._model is None:
            return b""

        content = io.BytesIO()
        torch.save(
            {
                "state_dict": self._model.state_dict(),
            },
            content,
        )
        return content.getvalue()

    def get_predictions(self, inputs: torch.Tensor) -> Sequence[Sequence[float]]:
        """Get model predictions.

        Args:
            inputs: Preprocessed input tensors

        Returns:
            List of prediction scores for each input
        """
        with torch.no_grad():
            inputs = inputs.to(self._device)
            output = self._model(inputs)
            predictions = torch.softmax(output, dim=1)
            return predictions.cpu().numpy()

    @log_exceptions
    def _infer_results(self) -> None:
        """Background inference thread.

        This method runs in a separate thread and processes inference requests.
        """
        while self._running:
            # Process batched requests
            pass

    def infer(self, requests: Sequence[ObjectId]) -> Iterable[ResultProvider]:
        """Run inference on multiple requests.

        Args:
            requests: List of object IDs to infer

        Yields:
            ResultProvider objects containing inference results
        """
        if not self._running or self._model is None:
            return

        for i in range(0, len(requests), self.config.test_batch_size):
            batch = []
            for object_id in requests[i : i + self.config.test_batch_size]:
                obj = self.context.retriever.get_ml_data(object_id)
                assert obj is not None
                batch.append((object_id, self.preprocess(obj)))

            results = self._process_batch(batch)
            yield from results

    def _process_batch(
        self,
        batch: list[tuple[ObjectId, torch.Tensor]],
    ) -> Iterable[ResultProvider]:
        """Process a batch of inference requests.

        Args:
            batch: List of (object_id, preprocessed_data) tuples

        Returns:
            List of ResultProvider objects
        """
        results = []
        with self._model_lock:
            tensors = torch.stack([f[1] for f in batch])
            predictions = self.get_predictions(tensors)

            for i in range(len(batch)):
                score = predictions[i]
                result_object = batch[i][0]

                # Create detection objects
                bboxes = [
                    Detection(class_name=class_name, confidence=float(score[class_idx]))
                    for class_idx, class_name in enumerate(self.context.class_list)
                ]

                results.append(
                    ResultProvider(
                        result_object,
                        sum(score),
                        bboxes,
                        self.version,
                    )
                )

        return results

    def stop(self) -> None:
        """Stop the model and release resources."""
        logger.info(f"Stopping model of version {self.version}")
        with self._model_lock:
            self._running = False
            self._model = None
```

**Key methods to implement**:

- `load_model(model_path)`: Load model architecture and weights
- `preprocess(obj)`: Preprocess input data for inference
- `get_predictions(inputs)`: Run inference and return predictions
- `infer(requests)`: Main inference entry point
- `_process_batch(batch)`: Process batched inference requests
- `serialize()`: Serialize model for transmission
- `stop()`: Clean up resources

## Model Context

The `ModelContext` object provides access to mission information:

```python
context.retriever: Retriever  # Data retrieval interface
context.model_dir: Path       # Directory for models
context.class_list: list[str]  # List of class labels
context.model_path(version, template="model-{}.pt")  # Get model file path
context.start_time: float      # Mission start time
context.model_input_queue     # Queue for inference requests
context.model_output_queue    # Queue for results
```

## Configuration Usage

Users configure your trainer in their mission YAML:

```yaml
train_strategy:
  type: your_trainer_name
  mode: hawk
  train_batch_size: 32
  learning_rate: 0.001
  # Your custom parameters
  param1: custom_value
```

## Testing Your Trainer

1. **Install dependencies**: Ensure all required packages are in `pyproject.toml` or installed via pip

2. **Build the package**:
   ```bash
   uv build
   ```

3. **Install locally**:
   ```bash
   pip install dist/*.whl
   ```

4. **Run tests**:
   ```bash
   pytest tests/ -m scout
   ```

## Examples

See existing trainers for reference:
- `dnn_classifier`: Standard DNN classifier trainer
- `fsl`: Few-shot learning trainer

## Common Patterns

### For Image Classifiers:
- Use PyTorch for model definition
- Implement preprocessing for images
- Return probability scores for each class

### For Object Detectors:
- Return bounding boxes with class names
- Include detection confidence scores
- May need custom serialization format

### For Few-Shot Learning:
- Use support sets for meta-learning
- May need specialized data augmentation
- Return feature vectors for clustering

## Troubleshooting

**ImportError: Unknown model**: Check that your entry point is correct in `pyproject.toml`

**Model not loading**: Verify `load_model` returns a valid model instance

**Training fails**: Check that `train_model` creates valid model files at the expected paths

**Inference errors**: Ensure `preprocess` and `get_predictions` handle input/output correctly
