# If you get a ModuleNotFound error install koinapy with `pip install koinapy`.
"""Koina model access with robust error handling, timeouts, and retry logic."""

import time
import logging
import socket
from typing import Optional, Tuple, List
from functools import wraps

import pandas as pd

logger = logging.getLogger(__name__)

# Default Koina settings
DEFAULT_KOINA_HOST = "koina.wilhelmlab.org:443"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds, will use exponential backoff

# Global flag to disable Koina entirely
_koina_disabled = False


def disable_koina():
    """Globally disable Koina predictions. All calls will raise KoinaDisabledError."""
    global _koina_disabled
    _koina_disabled = True
    logger.info("Koina predictions globally disabled")


def enable_koina():
    """Re-enable Koina predictions."""
    global _koina_disabled
    _koina_disabled = False
    logger.info("Koina predictions enabled")


def is_koina_disabled() -> bool:
    """Check if Koina is globally disabled."""
    return _koina_disabled


class KoinaError(Exception):
    """Base exception for Koina-related errors."""
    pass


class KoinaDisabledError(KoinaError):
    """Raised when Koina is disabled but a prediction is attempted."""
    pass


class KoinaConnectionError(KoinaError):
    """Raised when unable to connect to Koina server."""
    pass


class KoinaTimeoutError(KoinaError):
    """Raised when Koina request times out."""
    pass


class KoinaPredictionError(KoinaError):
    """Raised when Koina prediction fails."""

    def __init__(self, message: str, model_name: str = None, batch_size: int = None,
                 sample_peptides: List[str] = None):
        super().__init__(message)
        self.model_name = model_name
        self.batch_size = batch_size
        self.sample_peptides = sample_peptides or []

    def __str__(self):
        details = [super().__str__()]
        if self.model_name:
            details.append(f"Model: {self.model_name}")
        if self.batch_size:
            details.append(f"Batch size: {self.batch_size}")
        if self.sample_peptides:
            details.append(f"Sample peptides: {self.sample_peptides[:5]}")
        return " | ".join(details)


def check_koina_server(host: str = DEFAULT_KOINA_HOST, timeout: float = 5.0) -> Tuple[bool, Optional[str]]:
    """
    Check if Koina server is reachable.

    Args:
        host: Koina server host (format: "hostname:port")
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        # Parse host and port
        if ":" in host:
            hostname, port_str = host.rsplit(":", 1)
            port = int(port_str)
        else:
            hostname = host
            port = 443

        # Try to establish a socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()

        if result == 0:
            logger.debug(f"Koina server {host} is reachable")
            return True, None
        else:
            error_msg = f"Koina server {host} is not reachable (connection refused)"
            logger.warning(error_msg)
            return False, error_msg

    except socket.timeout:
        error_msg = f"Koina server {host} connection timed out after {timeout}s"
        logger.warning(error_msg)
        return False, error_msg
    except socket.gaierror as e:
        error_msg = f"Koina server {host} DNS resolution failed: {e}"
        logger.warning(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Koina server {host} connection check failed: {e}"
        logger.warning(error_msg)
        return False, error_msg


def retry_with_backoff(max_retries: int = DEFAULT_MAX_RETRIES,
                       initial_delay: float = DEFAULT_RETRY_DELAY,
                       backoff_factor: float = 2.0,
                       exceptions: tuple = (Exception,)):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Koina request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"Koina request failed after {max_retries + 1} attempts: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


class ModelFromKoina:
    """
    Class to access peptide property prediction models hosted on Koina.

    Features:
    - Server availability checking
    - Configurable timeouts
    - Automatic retry with exponential backoff
    - Detailed error logging

    Available properties include:
    - Fragment Intensity
    - Retention Time
    - Collision Cross Section
    - Cross-Linking Fragment Intensity
    - Flyability
    """

    def __init__(
        self,
        model_name: str,
        host: str = DEFAULT_KOINA_HOST,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        check_server: bool = True,
    ):
        """
        Args:
            model_name: Name of the Koina model. See https://koina.wilhelmlab.org/docs
            host: Host of the Koina server.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Initial delay between retries (uses exponential backoff).
            check_server: Whether to check server availability on initialization.

        Raises:
            KoinaDisabledError: If Koina is globally disabled.
            KoinaConnectionError: If server is not reachable and check_server=True.
        """
        if is_koina_disabled():
            raise KoinaDisabledError(
                "Koina predictions are disabled. Use enable_koina() to re-enable."
            )

        self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Check server availability if requested
        if check_server:
            available, error = check_koina_server(host, timeout=5.0)
            if not available:
                raise KoinaConnectionError(
                    f"Koina server not available: {error}. "
                    f"Use check_server=False to skip this check."
                )

        # Import koinapy here to allow graceful failure if not installed
        try:
            from koinapy import Koina
        except ImportError:
            raise ImportError(
                "koinapy is required for Koina predictions. "
                "Install with: pip install koinapy"
            )

        self.model = Koina(model_name, host)
        logger.info(f"Initialized Koina model: {model_name} @ {host}")

    @property
    def model_inputs(self):
        """Get required model inputs."""
        return self.model.model_inputs

    def predict(
        self,
        inputs: pd.DataFrame,
        filter_inputs: bool = True,
        raise_on_empty: bool = False,
    ) -> pd.DataFrame:
        """
        Predict the output of the model with retry logic.

        Args:
            inputs: Input data for the model. Requires a DataFrame with columns:
                - 'peptide_sequences': Peptide sequence with unimod format (always required)
                - 'precursor_charges': Precursor charge state (required for some models)
                - 'collision_energies': Collision energy (required for some models)
                - 'fragmentation_types': Fragmentation type (required for some models)
            filter_inputs: Whether to filter inputs based on model requirements.
            raise_on_empty: Whether to raise an error if all inputs are filtered out.

        Returns:
            pd.DataFrame: Output data from the model.

        Raises:
            KoinaDisabledError: If Koina is globally disabled.
            KoinaPredictionError: If prediction fails after all retries.
            ValueError: If required columns are missing or all inputs filtered.
        """
        if is_koina_disabled():
            raise KoinaDisabledError(
                "Koina predictions are disabled. Use enable_koina() to re-enable."
            )

        # Check if required columns are present
        required_columns = self.model.model_inputs.keys()
        for col in required_columns:
            if col not in inputs.columns:
                raise ValueError(
                    f"Input DataFrame is missing required column: {col}. "
                    f"Required columns: {list(required_columns)}"
                )

        original_count = len(inputs)

        # Filter inputs based on model requirements
        if filter_inputs:
            from .input_filters import filter_input_by_model
            inputs = filter_input_by_model(self.model_name, inputs)

        filtered_count = len(inputs)

        if filtered_count == 0:
            msg = (
                f"All {original_count} peptides were filtered out for model {self.model_name}. "
                f"Check peptide length, modifications, and charge states."
            )
            if raise_on_empty:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return pd.DataFrame()

        if filtered_count < original_count:
            logger.info(
                f"Filtered {original_count - filtered_count}/{original_count} peptides "
                f"for model {self.model_name}"
            )

        # Perform prediction with retry logic
        return self._predict_with_retry(inputs)

    def _predict_with_retry(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Internal method that performs prediction with retry logic."""
        delay = self.retry_delay
        last_exception = None
        sample_peptides = inputs["peptide_sequences"].head(5).tolist() if "peptide_sequences" in inputs.columns else []

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Koina prediction attempt {attempt + 1}/{self.max_retries + 1} "
                    f"for {len(inputs)} peptides"
                )
                predictions = self.model.predict(inputs)
                logger.debug(f"Koina prediction successful for {len(inputs)} peptides")
                return predictions

            except Exception as e:
                last_exception = e
                error_type = type(e).__name__

                if attempt < self.max_retries:
                    logger.warning(
                        f"Koina prediction failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{error_type}: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2.0  # Exponential backoff
                else:
                    logger.error(
                        f"Koina prediction failed after {self.max_retries + 1} attempts. "
                        f"Model: {self.model_name}, Batch size: {len(inputs)}, "
                        f"Sample peptides: {sample_peptides}"
                    )

        # Raise detailed error after all retries exhausted
        raise KoinaPredictionError(
            str(last_exception),
            model_name=self.model_name,
            batch_size=len(inputs),
            sample_peptides=sample_peptides,
        )

    def __repr__(self):
        return (
            f"ModelFromKoina(model_name='{self.model_name}', host='{self.host}', "
            f"timeout={self.timeout}, max_retries={self.max_retries})"
        )
