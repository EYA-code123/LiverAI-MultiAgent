# =============================================================================
# LiverAI-MultiAgent
# TUMOR CLASSIFICATION AGENT
# =============================================================================

import os
import time
import traceback

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


class TumorClassificationAgent:
    """
    Tumor Classification Agent.

    Input:
        MRI 2D image.

    Output:
        5-class liver pathology classification.

    Expected model:
        Keras model (.keras)

    Dataset information used by the project:
        - MRI 2D slices
        - 5 classes
        - Recommended image size: 224x224
    """

    DEFAULT_CLASSES = [
        "Angiosarcoma",
        "Cholangiocarcinoma",
        "Healthy",
        "Hemangioma",
        "Hepatocellular Carcinoma",
    ]

    def __init__(
        self,
        model_path,
        class_names=None,
        image_size=(224, 224),
        channels=3,
    ):
        self.agent_id = "TumorClassificationAgent"
        self.agent = self.agent_id
        self.task_type = "tumor_classification"

        self.model_path = model_path
        self.image_size = tuple(image_size)
        self.channels = channels

        self.class_names = (
            list(class_names)
            if class_names is not None
            else self.DEFAULT_CLASSES.copy()
        )

        self.model = None

        self._validate_environment()
        self._load_model()

    # =========================================================================
    # ENVIRONMENT
    # =========================================================================

    def _validate_environment(self):

        if tf is None:
            raise ImportError(
                "TensorFlow is required for TumorClassificationAgent."
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Tumor model not found:\n{self.model_path}"
            )

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    def _load_model(self):

        print("=" * 70)
        print("TUMOR CLASSIFICATION AGENT")
        print("=" * 70)

        print("Model:", self.model_path)

        try:

            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False,
            )

            print("✓ Tumor Keras model loaded")

            try:
                print("Input shape :", self.model.input_shape)
            except Exception:
                pass

            try:
                print("Output shape:", self.model.output_shape)
            except Exception:
                pass

        except Exception as e:

            print("✗ Failed to load tumor model")
            print("Error:", e)

            raise

    # =========================================================================
    # INPUT LOADING
    # =========================================================================

    def _load_image(self, image):

        if image is None:
            raise ValueError(
                "Tumor agent received no MRI image."
            )

        # ---------------------------------------------------------------------
        # PATH
        # ---------------------------------------------------------------------

        if isinstance(image, (str, os.PathLike)):

            if not os.path.exists(image):
                raise FileNotFoundError(
                    f"MRI image not found: {image}"
                )

            if Image is None:
                raise ImportError(
                    "Pillow is required to load image files."
                )

            image = Image.open(image)

        # ---------------------------------------------------------------------
        # PIL IMAGE
        # ---------------------------------------------------------------------

        if Image is not None and isinstance(image, Image.Image):

            image = image.convert("RGB")

            image = image.resize(
                self.image_size
            )

            image = np.asarray(
                image,
                dtype=np.float32
            )

        # ---------------------------------------------------------------------
        # NUMPY ARRAY
        # ---------------------------------------------------------------------

        elif isinstance(image, np.ndarray):

            image = image.astype(
                np.float32,
                copy=False
            )

            # Remove singleton dimensions
            image = np.squeeze(image)

            # -------------------------------------------------------------
            # Grayscale
            # -------------------------------------------------------------

            if image.ndim == 2:

                image = self._normalize_image(
                    image
                )

                if Image is not None:

                    pil = Image.fromarray(
                        (image * 255).astype(np.uint8)
                    )

                    pil = pil.resize(
                        self.image_size
                    )

                    image = np.asarray(
                        pil,
                        dtype=np.float32
                    )

                else:

                    image = self._resize_numpy(
                        image,
                        self.image_size
                    )

                image = np.stack(
                    [image, image, image],
                    axis=-1
                )

            # -------------------------------------------------------------
            # H x W x 1
            # -------------------------------------------------------------

            elif image.ndim == 3 and image.shape[-1] == 1:

                image = np.repeat(
                    image,
                    3,
                    axis=-1
                )

                image = self._resize_numpy(
                    image,
                    self.image_size
                )

            # -------------------------------------------------------------
            # H x W x 3
            # -------------------------------------------------------------

            elif image.ndim == 3 and image.shape[-1] == 3:

                image = self._resize_numpy(
                    image,
                    self.image_size
                )

            else:

                raise ValueError(
                    "Unsupported MRI array shape: "
                    f"{image.shape}"
                )

        else:

            raise TypeError(
                "MRI input must be a file path, PIL image "
                "or numpy array."
            )

        # ---------------------------------------------------------------------
        # NORMALIZE
        # ---------------------------------------------------------------------

        image = self._normalize_image(
            image
        )

        # ---------------------------------------------------------------------
        # ENSURE CHANNELS
        # ---------------------------------------------------------------------

        if image.ndim == 2:

            image = np.stack(
                [image] * 3,
                axis=-1
            )

        if image.ndim != 3:

            raise ValueError(
                f"Final MRI image must be 3D. "
                f"Got shape={image.shape}"
            )

        if image.shape[-1] == 1:

            image = np.repeat(
                image,
                3,
                axis=-1
            )

        if image.shape[-1] != 3:

            raise ValueError(
                f"Expected 3 channels, got {image.shape[-1]}"
            )

        return image.astype(
            np.float32
        )

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_image(image):

        image = np.asarray(
            image,
            dtype=np.float32
        )

        if not np.isfinite(image).all():

            image = np.nan_to_num(
                image,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        min_value = float(
            image.min()
        )

        max_value = float(
            image.max()
        )

        if max_value > min_value:

            image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        else:

            image = np.zeros_like(
                image,
                dtype=np.float32
            )

        return image

    # =========================================================================
    # NUMPY RESIZE FALLBACK
    # =========================================================================

    @staticmethod
    def _resize_numpy(
        image,
        target_size
    ):

        target_h, target_w = target_size

        if image.ndim == 2:

            h, w = image.shape

            y_idx = np.linspace(
                0,
                h - 1,
                target_h
            ).astype(int)

            x_idx = np.linspace(
                0,
                w - 1,
                target_w
            ).astype(int)

            return image[
                np.ix_(
                    y_idx,
                    x_idx
                )
            ]

        if image.ndim == 3:

            h, w, c = image.shape

            y_idx = np.linspace(
                0,
                h - 1,
                target_h
            ).astype(int)

            x_idx = np.linspace(
                0,
                w - 1,
                target_w
            ).astype(int)

            return image[
                np.ix_(
                    y_idx,
                    x_idx,
                    np.arange(c)
                )
            ]

        raise ValueError(
            f"Unsupported image dimension: {image.ndim}"
        )

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, image):

        start_time = time.perf_counter()

        try:

            if self.model is None:
                raise RuntimeError(
                    "Tumor model is not loaded."
                )

            processed = self._load_image(
                image
            )

            batch = np.expand_dims(
                processed,
                axis=0
            )

            # -------------------------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------------------------

            raw_output = self.model.predict(
                batch,
                verbose=0
            )

            probabilities = self._extract_probabilities(
                raw_output
            )

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            confidence = float(
                probabilities[predicted_index]
            )

            if predicted_index < len(
                self.class_names
            ):

                prediction = self.class_names[
                    predicted_index
                ]

            else:

                prediction = str(
                    predicted_index
                )

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            uncertainty = float(
                max(
                    0.0,
                    min(
                        1.0,
                        1.0 - confidence
                    )
                )
            )

            # -------------------------------------------------------------
            # PROBABILITY DICTIONARY
            # -------------------------------------------------------------

            probability_dict = {}

            for i, probability in enumerate(
                probabilities
            ):

                if i < len(
                    self.class_names
                ):

                    label = self.class_names[i]

                else:

                    label = str(i)

                probability_dict[
                    label
                ] = float(probability)

            return {
                "agent_id":
                    self.agent_id,

                "agent":
                    self.agent_id,

                "task_type":
                    self.task_type,

                "status":
                    "completed",

                "prediction":
                    prediction,

                "predicted_class":
                    prediction,

                "predicted_index":
                    predicted_index,

                "probability":
                    confidence,

                "probabilities":
                    probability_dict,

                "confidence":
                    confidence,

                "uncertainty":
                    uncertainty,

                "quality":
                    1.0,

                "missing_data_ratio":
                    0.0,

                "latency_ms":
                    elapsed_ms,

                "details": {
                    "modality":
                        "MRI",

                    "input_type":
                        "2D image",

                    "image_size":
                        list(
                            self.image_size
                        ),

                    "num_classes":
                        len(
                            self.class_names
                        ),

                    "classes":
                        self.class_names,
                },

                "explanation":
                    (
                        "Tumor classification performed "
                        "using the trained Keras model."
                    ),

                "error":
                    None,
            }

        except Exception as e:

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000.0

            traceback.print_exc()

            return {
                "agent_id":
                    self.agent_id,

                "agent":
                    self.agent_id,

                "task_type":
                    self.task_type,

                "status":
                    "error",

                "prediction":
                    None,

                "probability":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "latency_ms":
                    elapsed_ms,

                "details":
                    {},

                "explanation":
                    None,

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc(),
            }

    # =========================================================================
    # OUTPUT HANDLING
    # =========================================================================

    @staticmethod
    def _extract_probabilities(
        raw_output
    ):

        if isinstance(
            raw_output,
            (list, tuple)
        ):

            raw_output = raw_output[0]

        probabilities = np.asarray(
            raw_output
        )

        probabilities = np.squeeze(
            probabilities
        )

        if probabilities.ndim != 1:

            raise ValueError(
                "Unexpected tumor model output shape: "
                f"{probabilities.shape}"
            )

        probabilities = probabilities.astype(
            np.float64
        )

        if not np.isfinite(
            probabilities
        ).all():

            raise ValueError(
                "Tumor model returned NaN/Inf probabilities."
            )

        # ---------------------------------------------------------------------
        # If output is not already a probability distribution,
        # convert logits to softmax.
        # ---------------------------------------------------------------------

        total = float(
            probabilities.sum()
        )

        if (
            np.any(probabilities < 0)
            or
            not np.isclose(
                total,
                1.0,
                atol=1e-3
            )
        ):

            probabilities = (
                np.exp(
                    probabilities
                    -
                    np.max(
                        probabilities
                    )
                )
            )

            probabilities = (
                probabilities
                /
                probabilities.sum()
            )

        return probabilities.astype(
            np.float32
        )

    # =========================================================================
    # MODEL INFORMATION
    # =========================================================================

    def get_model_info(self):

        info = {
            "agent":
                self.agent_id,

            "task":
                self.task_type,

            "model_path":
                self.model_path,

            "classes":
                self.class_names,

            "image_size":
                self.image_size,
        }

        if self.model is not None:

            try:
                info["input_shape"] = str(
                    self.model.input_shape
                )
            except Exception:
                pass

            try:
                info["output_shape"] = str(
                    self.model.output_shape
                )
            except Exception:
                pass

        return info

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        return {
            "agent":
                self.agent_id,

            "model_loaded":
                self.model is not None,

            "model_exists":
                os.path.exists(
                    self.model_path
                ),

            "status":
                (
                    "ready"
                    if self.model is not None
                    else "not_ready"
                ),
        }
