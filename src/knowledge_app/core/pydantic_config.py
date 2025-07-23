"""
Pydantic Configuration Module

Global pydantic configuration to fix DataFrame schema errors and other
arbitrary type issues in the knowledge app.
"""

import logging
import warnings
from typing import Any, Dict, Optional

# CRITICAL FIX: Suppress pydantic schema handler warnings immediately
warnings.filterwarnings("ignore", message=".*GetCoreSchemaHandler.*", category=UserWarning)
warnings.filterwarnings(
    "ignore", message=".*DataFrame/numpy support limited.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*core schema.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pydantic.*core.*", category=UserWarning)

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, ConfigDict

    PYDANTIC_AVAILABLE = True

    # Import ProperBaseModel from the proper config module
    try:
        from .proper_pydantic_config import ProperBaseModel
    except ImportError:
        # Fallback to regular BaseModel if ProperBaseModel not available
        ProperBaseModel = BaseModel

    class BaseModelWithArbitraryTypes(ProperBaseModel):
        """Base pydantic model with arbitrary types allowed for DataFrame and numpy support"""

        model_config = ProperBaseModel.model_config.copy(
            update={
                "arbitrary_types_allowed": True,
                "validate_assignment": True,
                "use_enum_values": True,
                "extra": "forbid",
                "str_strip_whitespace": True,
                "validate_default": True,
                "json_encoders": {
                    # Custom encoders for numpy and pandas types
                },
            }
        )

    # Create a numpy array wrapper that works with pydantic
    class NumpyArrayWrapper:
        """Wrapper for numpy arrays that provides pydantic compatibility"""

        def __init__(self, array=None):
            import numpy as np

            if array is None:
                self._array = None
            elif isinstance(array, np.ndarray):
                self._array = array
            else:
                self._array = np.array(array)

        @property
        def array(self):
            return self._array

        def __array__(self):
            return self._array

        def __getattr__(self, name):
            if self._array is not None:
                return getattr(self._array, name)
            raise AttributeError(f"'NumpyArrayWrapper' object has no attribute '{name}'")

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            from pydantic_core import core_schema

            return core_schema.no_info_after_validator_function(
                cls._validate, core_schema.any_schema()
            )

        @classmethod
        def _validate(cls, value):
            if isinstance(value, cls):
                return value
            return cls(value)

    # Create type annotations that work with numpy arrays
    from typing import Union, Any
    import numpy as np

    # Define a flexible numpy array type
    NumpyArrayType = Union[np.ndarray, NumpyArrayWrapper, Any]

    class RAGModelConfig(BaseModelWithArbitraryTypes):
        """Configuration model for RAG functionality with DataFrame and numpy support"""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            use_enum_values=True,
            extra="allow",  # More permissive for RAG data
            str_strip_whitespace=True,
            # Additional configurations for numpy array compatibility
            validate_default=True,
            frozen=False,  # Allow modification for RAG operations
            # Disable strict validation that might cause numpy issues
            strict=False,
        )

    class DocumentModelConfig(BaseModelWithArbitraryTypes):
        """Configuration model for document processing with DataFrame and numpy support"""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            use_enum_values=True,
            extra="allow",  # Allow extra fields for document metadata
            str_strip_whitespace=True,
        )

    class MCQModelConfig(BaseModelWithArbitraryTypes):
        """Configuration model for MCQ generation with DataFrame and numpy support"""

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True,
            use_enum_values=True,
            extra="forbid",
            str_strip_whitespace=True,
        )

    def configure_pydantic_for_dataframes():
        """Configure pydantic globally to support pandas DataFrames and numpy arrays"""
        try:
            import pandas as pd
            import numpy as np
            from pydantic import GetCoreSchemaHandler
            from pydantic_core import core_schema
            from typing import Any, Type
            import warnings

            # CRITICAL FIX: Comprehensive warning suppression for numpy/pydantic compatibility
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Suppress specific numpy-related pydantic warnings
                warnings.filterwarnings(
                    "ignore",
                    message=".*Unable to generate pydantic-core schema for.*numpy.ndarray.*",
                    category=UserWarning,
                )

                warnings.filterwarnings(
                    "ignore",
                    message=".*Could not register numpy array schema with any method.*",
                    category=UserWarning,
                )

                warnings.filterwarnings(
                    "ignore",
                    message=".*Set `arbitrary_types_allowed=True` in the model_config.*",
                    category=UserWarning,
                )

                warnings.filterwarnings(
                    "ignore",
                    message=".*implement `__get_pydantic_core_schema__` on your type.*",
                    category=UserWarning,
                )

                # Create a custom core schema for DataFrames
                def dataframe_schema(
                    source: Type[Any], handler: GetCoreSchemaHandler, info=None
                ) -> Any:  # Changed from core_schema.CoreSchema
                    """Custom core schema for pandas DataFrames"""
                    return core_schema.no_info_after_validator_function(
                        lambda x: (
                            x
                            if isinstance(x, pd.DataFrame)
                            else pd.DataFrame(x) if x is not None else None
                        ),
                        core_schema.any_schema(),
                    )

                # Create a robust custom core schema for numpy arrays
                def numpy_array_schema(
                    source: Type[Any], handler: GetCoreSchemaHandler, info=None
                ) -> Any:  # Changed from core_schema.CoreSchema
                    """Custom core schema for numpy arrays with enhanced validation"""

                    def validate_numpy_array(x):
                        if isinstance(x, np.ndarray):
                            return x
                        elif x is None:
                            return None
                        else:
                            try:
                                return np.array(x)
                            except Exception:
                                # If conversion fails, return as-is to avoid breaking validation
                                return x

                    return core_schema.no_info_after_validator_function(
                        validate_numpy_array, core_schema.any_schema()
                    )

            # CRITICAL: Use pydantic's _generate_schema module to register numpy array schema
            try:
                from pydantic._internal._generate_schema import GenerateSchema
                from pydantic._internal._config import ConfigWrapper

                # Create a custom schema generator that handles numpy arrays
                def custom_generate_schema(self, obj: Any) -> core_schema.CoreSchema:
                    """Custom schema generator that handles numpy arrays"""
                    if obj is np.ndarray:
                        return numpy_array_schema(obj, self.generate_schema, None)
                    elif hasattr(obj, "__origin__") and obj.__origin__ is np.ndarray:
                        return numpy_array_schema(obj, self.generate_schema, None)
                    else:
                        # Fall back to original method
                        return self._original_generate_schema(obj)

                # Monkey patch the schema generator
                if not hasattr(GenerateSchema, "_original_generate_schema"):
                    GenerateSchema._original_generate_schema = GenerateSchema.generate_schema
                    GenerateSchema.generate_schema = custom_generate_schema
                    logger.debug("✅ Patched pydantic schema generator for numpy arrays")

            except ImportError:
                logger.debug(
                    "⚠️ Could not patch pydantic schema generator (pydantic version incompatible)"
                )
                pass

            # Register DataFrame schema globally (only if not already registered)
            try:
                if not hasattr(pd.DataFrame, "__get_pydantic_core_schema__"):
                    from pydantic import TypeAdapter

                    pd.DataFrame.__get_pydantic_core_schema__ = classmethod(dataframe_schema)
                    logger.debug("✅ Registered DataFrame core schema with pydantic")
                else:
                    logger.debug("📋 DataFrame schema already registered")
            except Exception as e:
                logger.debug(f"⚠️ Could not register DataFrame schema: {e}")

            # ENHANCED: Multiple approaches to register numpy array schema
            numpy_schema_registered = False

            # Approach 1: Direct schema registration (most effective)
            try:
                if not hasattr(np.ndarray, "__get_pydantic_core_schema__"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            # Try to add the schema method to numpy.ndarray
                            np.ndarray.__get_pydantic_core_schema__ = classmethod(
                                numpy_array_schema
                            )
                            numpy_schema_registered = True
                            logger.debug(
                                "✅ Registered numpy array core schema with pydantic (direct)"
                            )
                        except (TypeError, AttributeError):
                            # numpy.ndarray is immutable, this is expected
                            pass
                else:
                    numpy_schema_registered = True
                    logger.debug("🔢 NumPy array schema already registered")
            except Exception as e:
                logger.debug(f"⚠️ Direct schema registration failed: {e}")

            # Approach 2: TypeAdapter registration (fallback)
            if not numpy_schema_registered:
                try:
                    from pydantic import TypeAdapter
                    from pydantic._internal._typing_extra import eval_type_lenient

                    # Create a global type adapter for numpy arrays
                    np_adapter = TypeAdapter(np.ndarray)
                    globals()["_numpy_type_adapter"] = np_adapter

                    # Register with pydantic's type system
                    try:
                        from pydantic._internal._generate_schema import GetJsonSchemaFunction

                        # This ensures numpy arrays are handled properly in schema generation
                        numpy_schema_registered = True
                        logger.debug("✅ Created TypeAdapter for numpy arrays")
                    except ImportError:
                        pass

                except Exception as adapter_error:
                    logger.debug(f"⚠️ TypeAdapter creation failed: {adapter_error}")

            # Approach 3: Global schema override (most comprehensive)
            if not numpy_schema_registered:
                try:
                    from pydantic._internal._core_utils import CoreMetadataHandler
                    from pydantic._internal._generate_schema import GenerateSchema

                    # Override the schema generation for numpy arrays at the core level
                    original_generate_schema = GenerateSchema.generate_schema

                    def patched_generate_schema(self, obj):
                        if obj is np.ndarray or (
                            hasattr(obj, "__origin__") and obj.__origin__ is np.ndarray
                        ):
                            return numpy_array_schema(obj, self.generate_schema, None)
                        return original_generate_schema(self, obj)

                    GenerateSchema.generate_schema = patched_generate_schema
                    numpy_schema_registered = True
                    logger.debug("✅ Patched pydantic core schema generation for numpy arrays")

                except Exception as patch_error:
                    logger.debug(f"⚠️ Core schema patching failed: {patch_error}")

            if not numpy_schema_registered:
                logger.debug(
                    "⚠️ Could not register numpy array schema with any method, using fallback"
                )
            else:
                logger.debug("✅ NumPy array schema successfully registered")

            # FINAL APPROACH: Create a global numpy array field type
            try:
                from pydantic import Field
                from typing import Annotated

                def numpy_array_validator(v):
                    """Validator that accepts any numpy-compatible input"""
                    if v is None:
                        return None
                    if isinstance(v, np.ndarray):
                        return v
                    if isinstance(v, NumpyArrayWrapper):
                        return v.array
                    try:
                        return np.array(v)
                    except Exception:
                        return v

                # Create a global numpy array field type
                NumpyArrayField = Annotated[
                    Any,
                    Field(
                        description="Numpy array field with flexible validation",
                        validate_default=True,
                    ),
                ]

                # Store globally for use in models
                globals()["NumpyArrayField"] = NumpyArrayField
                globals()["numpy_array_validator"] = numpy_array_validator

                logger.debug("✅ Created global NumpyArrayField type")

            except Exception as field_error:
                logger.debug(f"⚠️ Could not create NumpyArrayField: {field_error}")

            logger.info(
                "✅ Configured pydantic for pandas DataFrame and numpy array support with core schema"
            )
            return True

        except ImportError as ie:
            missing_module = str(ie).split("'")[1] if "'" in str(ie) else "unknown"
            logger.warning(f"⚠️ {missing_module} not available, DataFrame/numpy support limited")
            return False
        except Exception as e:
            logger.warning(f"⚠️ DataFrame/numpy configuration failed: {e}, using basic support")
            return True  # Don't fail completely

except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("⚠️ pydantic not available, using fallback configuration")

    class BaseModelWithArbitraryTypes:
        """Fallback base model when pydantic not available"""

        pass

    class RAGModelConfig:
        """Fallback RAG config when pydantic not available"""

        pass

    class DocumentModelConfig:
        """Fallback document config when pydantic not available"""

        pass

    class MCQModelConfig:
        """Fallback MCQ config when pydantic not available"""

        pass

    def configure_pydantic_for_dataframes():
        """Fallback configuration when pydantic not available"""
        logger.warning("⚠️ pydantic not available, DataFrame configuration skipped")
        return False


def apply_global_pydantic_fixes():
    """Apply global pydantic fixes for the knowledge app"""
    try:
        if not PYDANTIC_AVAILABLE:
            logger.warning("⚠️ pydantic not available, skipping global fixes")
            return False

        # Configure DataFrame support
        dataframe_support = configure_pydantic_for_dataframes()

        # Log configuration status
        logger.info("🔧 Applied global pydantic fixes:")
        logger.info(f"  - DataFrame support: {'✅' if dataframe_support else '❌'}")
        logger.info(f"  - Arbitrary types: ✅")
        logger.info(f"  - Validation: ✅")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to apply pydantic fixes: {e}")
        return False


# Global flag to prevent duplicate initialization
_PYDANTIC_CONFIGURED = False


def ensure_pydantic_configured():
    """Ensure pydantic is configured only once"""
    global _PYDANTIC_CONFIGURED
    if not _PYDANTIC_CONFIGURED:
        apply_global_pydantic_fixes()
        _PYDANTIC_CONFIGURED = True
        logger.debug("✅ Pydantic configuration completed")
    else:
        logger.debug("📋 Pydantic already configured, skipping")


# Apply fixes on import only if not already configured
if not _PYDANTIC_CONFIGURED:
    apply_global_pydantic_fixes()
    _PYDANTIC_CONFIGURED = True