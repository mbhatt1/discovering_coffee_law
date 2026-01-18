"""
Internal States Experiment for Open-Weight Models.

This experiment implements "Open-weight internal-state replication" for the
COFFEE Law validation paper, addressing the "embeddings are proxies" critique.

By extracting internal states (residual stream, attention outputs, value vectors)
at different layers of open-weight models (Llama, Mistral), we can show that
the OU dynamics observed in embeddings are not artifacts but reflect genuine
internal representations.
"""

from typing import Optional, Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from openai import OpenAI

from .base import BaseExperiment, ExperimentResult
from ..utils.embeddings import EmbeddingClient
from ..utils.math import fit_power_law, PowerLawFit
from ..config import ExperimentConfig

# Optional imports for HuggingFace transformers
_TRANSFORMERS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class InternalStatesConfig:
    """Configuration for internal states experiment."""
    model_name: str = "meta-llama/Llama-3.2-1B"  # Default to 1B for feasibility
    num_continuations: int = 20
    max_tokens: int = 100
    sample_positions: List[int] = field(default_factory=lambda: [10, 20, 30, 50, 70, 100])
    temperature: float = 1.0
    layers_to_analyze: Optional[List[int]] = None  # None = all layers
    state_types: List[str] = field(default_factory=lambda: [
        "residual_pre",    # Hidden state before LayerNorm
        "residual_post",   # Hidden state after LayerNorm
        "attention_out",   # Output of attention mechanism
    ])
    device: Optional[str] = None  # None = auto-detect
    dtype: str = "float16"  # float16, bfloat16, or float32
    use_flash_attention: bool = False  # Disabled by default for compatibility


@dataclass
class LayerActivations:
    """Container for activations from a single layer."""
    layer_idx: int
    residual_pre: Optional[np.ndarray] = None
    residual_post: Optional[np.ndarray] = None
    attention_out: Optional[np.ndarray] = None
    value_vectors: Optional[np.ndarray] = None


class InternalStatesExperiment(BaseExperiment):
    """
    Experiment to extract and analyze internal states from open-weight models.

    This addresses the "embeddings are proxies" critique by showing that
    internal representations at various layers exhibit the same OU dynamics
    as external embeddings.

    Key features:
    - Supports Llama and Mistral models via HuggingFace transformers
    - Extracts residual stream states (pre/post LayerNorm)
    - Captures attention outputs and value vectors
    - Computes variance-vs-position for each layer type
    - Estimates Hurst exponent per layer and state type
    """

    experiment_name = "internal_states"

    # Default models - focused on 1B for feasibility
    SUPPORTED_MODELS = {
        "llama-1b": "meta-llama/Llama-3.2-1B",
        "llama-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }

    def __init__(
        self,
        config: ExperimentConfig,
        client: Optional[OpenAI] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        internal_config: Optional[InternalStatesConfig] = None,
        quick_mode: bool = False
    ):
        """
        Initialize the internal states experiment.

        Args:
            config: Base experiment configuration
            client: Optional OpenAI client for comparison embeddings
            embedding_client: Optional embedding client
            internal_config: Configuration specific to internal states extraction
            quick_mode: If True, use reduced parameters for faster testing
        """
        super().__init__(config, client, embedding_client)

        if quick_mode:
            self.internal_config = internal_config or InternalStatesConfig(
                num_continuations=5,
                max_tokens=50,
                sample_positions=[10, 25, 50],
            )
        else:
            self.internal_config = internal_config or InternalStatesConfig()

        # Model and tokenizer (loaded lazily)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._hooks = []
        self._activations: Dict[str, List[torch.Tensor]] = {}

        # Results storage
        self._layer_results: Dict[str, Dict[int, Dict]] = {}
        self._comparison_results: Dict[str, Any] = {}

    def _check_dependencies(self) -> Tuple[bool, str]:
        """Check if required dependencies are available."""
        if not _TORCH_AVAILABLE:
            return False, "PyTorch is required but not installed. Install with: pip install torch"
        if not _TRANSFORMERS_AVAILABLE:
            return False, "Transformers is required but not installed. Install with: pip install transformers"
        return True, ""

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if self.internal_config.device:
            return self.internal_config.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_dtype(self) -> "torch.dtype":
        """Get the torch dtype from config."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.internal_config.dtype, torch.float16)

    def _load_model(self) -> None:
        """Load the specified model and tokenizer."""
        if self._model is not None:
            return

        model_name = self.internal_config.model_name

        # Handle model aliases
        if model_name in self.SUPPORTED_MODELS:
            model_name = self.SUPPORTED_MODELS[model_name]

        self.log(f"Loading model: {model_name}")

        self._device = self._detect_device()
        dtype = self._get_dtype()

        self.log(f"Using device: {self._device}, dtype: {dtype}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Set pad token if not present
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with appropriate settings
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto" if self._device == "cuda" else None,
            "trust_remote_code": True,
        }

        # Enable flash attention if available and requested
        if self.internal_config.use_flash_attention and self._device == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                self.log("Flash attention not available, using default attention")

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Move to device if not using device_map
        if self._device != "cuda":
            self._model = self._model.to(self._device)

        self._model.eval()
        self.log(f"Model loaded successfully. Num layers: {self._get_num_layers()}")

    def _get_num_layers(self) -> int:
        """Get the number of transformer layers in the model."""
        if hasattr(self._model, "config"):
            return getattr(self._model.config, "num_hidden_layers", 32)
        return 32

    def _get_layers_to_analyze(self) -> List[int]:
        """Get the list of layer indices to analyze."""
        num_layers = self._get_num_layers()

        if self.internal_config.layers_to_analyze is not None:
            return [l for l in self.internal_config.layers_to_analyze if 0 <= l < num_layers]

        # Default: sample layers across the depth
        if num_layers <= 8:
            return list(range(num_layers))
        else:
            # Sample early, middle, and late layers
            return [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]

    def _register_hooks(self, layers: List[int]) -> None:
        """Register forward hooks to capture activations."""
        self._clear_hooks()
        self._activations = {
            "residual_pre": [],
            "residual_post": [],
            "attention_out": [],
            "value_vectors": [],
        }

        model_layers = self._get_model_layers()

        for layer_idx in layers:
            if layer_idx >= len(model_layers):
                continue

            layer = model_layers[layer_idx]

            # Hook for residual stream before layer norm
            if hasattr(layer, "input_layernorm"):
                hook = layer.input_layernorm.register_forward_hook(
                    self._create_activation_hook("residual_pre", layer_idx)
                )
                self._hooks.append(hook)

            # Hook for attention output
            if hasattr(layer, "self_attn"):
                hook = layer.self_attn.register_forward_hook(
                    self._create_attention_hook("attention_out", layer_idx)
                )
                self._hooks.append(hook)

            # Hook for post-attention layernorm (residual_post)
            if hasattr(layer, "post_attention_layernorm"):
                hook = layer.post_attention_layernorm.register_forward_hook(
                    self._create_activation_hook("residual_post", layer_idx)
                )
                self._hooks.append(hook)

    def _get_model_layers(self) -> list:
        """Get the list of transformer layers from the model."""
        if hasattr(self._model, "model"):
            # Llama-style models
            if hasattr(self._model.model, "layers"):
                return list(self._model.model.layers)
        if hasattr(self._model, "transformer"):
            # GPT-style models
            if hasattr(self._model.transformer, "h"):
                return list(self._model.transformer.h)
        return []

    def _create_activation_hook(
        self,
        activation_type: str,
        layer_idx: int
    ) -> Callable:
        """Create a hook function to capture activations."""
        def hook(module, input, output):
            # Input to layernorm is the residual stream
            if isinstance(input, tuple):
                tensor = input[0]
            else:
                tensor = input

            if tensor is not None:
                # Store as (layer_idx, activation)
                self._activations[activation_type].append(
                    (layer_idx, tensor.detach().cpu())
                )
        return hook

    def _create_attention_hook(
        self,
        activation_type: str,
        layer_idx: int
    ) -> Callable:
        """Create a hook function to capture attention outputs."""
        def hook(module, input, output):
            # Attention output is typically the first element
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            if tensor is not None:
                self._activations[activation_type].append(
                    (layer_idx, tensor.detach().cpu())
                )
        return hook

    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    def _extract_internal_states(
        self,
        text: str,
        layers: List[int]
    ) -> Dict[int, LayerActivations]:
        """
        Extract internal states at specified layers for given text.

        Args:
            text: Input text to process
            layers: Layer indices to extract states from

        Returns:
            Dictionary mapping layer index to LayerActivations
        """
        # Register hooks
        self._register_hooks(layers)

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move to device
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            _ = self._model(**inputs, output_hidden_states=True)

        # Organize activations by layer
        layer_activations = {}

        for layer_idx in layers:
            activations = LayerActivations(layer_idx=layer_idx)

            # Extract activations for this layer
            for act_type in ["residual_pre", "residual_post", "attention_out"]:
                for stored_layer, tensor in self._activations.get(act_type, []):
                    if stored_layer == layer_idx:
                        # Take the last token's activation as representative
                        if tensor.dim() == 3:  # (batch, seq, hidden)
                            act = tensor[0, -1, :].numpy()
                        else:
                            act = tensor.flatten().numpy()
                        setattr(activations, act_type, act)
                        break

            layer_activations[layer_idx] = activations

        self._clear_hooks()
        return layer_activations

    def _generate_continuations_local(
        self,
        prefix: str,
        num_continuations: int,
        max_tokens: int,
        temperature: float
    ) -> List[str]:
        """Generate continuations using the local model."""
        continuations = []

        inputs = self._tokenizer(prefix, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        for i in range(num_continuations):
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    top_p=0.95,
                )

            # Decode only the new tokens
            input_length = inputs["input_ids"].shape[1]
            continuation_ids = outputs[0, input_length:]
            continuation = self._tokenizer.decode(continuation_ids, skip_special_tokens=True)
            continuations.append(continuation)

            if self.config.output.verbose and (i + 1) % 5 == 0:
                self.log(f"Generated {i + 1}/{num_continuations} continuations")

        return continuations

    def _compute_layer_variance(
        self,
        activations_by_position: Dict[int, List[np.ndarray]],
        state_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute variance of internal states at each position.

        Args:
            activations_by_position: Dict mapping position to list of activation vectors
            state_type: Type of state being analyzed (for logging)

        Returns:
            (positions, variances) arrays
        """
        positions = []
        variances = []

        for pos in sorted(activations_by_position.keys()):
            acts = activations_by_position[pos]
            if len(acts) < 2:
                continue

            # Stack into matrix
            act_matrix = np.array(acts)

            # Compute centroid
            centroid = np.mean(act_matrix, axis=0)

            # Mean squared L2 distance from centroid
            distances_sq = np.sum((act_matrix - centroid) ** 2, axis=1)
            var = np.mean(distances_sq)

            if var > 1e-10:
                positions.append(pos)
                variances.append(var)

        return np.array(positions), np.array(variances)

    def _analyze_layer_dynamics(
        self,
        layer_idx: int,
        state_type: str,
        positions: np.ndarray,
        variances: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze OU dynamics for a specific layer and state type.

        Returns metrics including Hurst exponent and fit quality.
        """
        if len(positions) < 3:
            return {
                "valid": False,
                "error": f"Not enough data points: {len(positions)}"
            }

        try:
            fit_result = fit_power_law(positions, variances)
            hurst_exponent = fit_result.exponent / 2.0

            return {
                "valid": True,
                "hurst_exponent": float(hurst_exponent),
                "power_law_exponent": float(fit_result.exponent),
                "r_squared": float(fit_result.r_squared),
                "amplitude": float(fit_result.amplitude),
                "positions": positions.tolist(),
                "variances": variances.tolist(),
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }

    def run(self) -> ExperimentResult:
        """Execute the internal states experiment."""
        # Check dependencies
        deps_ok, deps_error = self._check_dependencies()
        if not deps_ok:
            self.log(f"Dependency check failed: {deps_error}")
            return ExperimentResult(
                experiment_name=self.experiment_name,
                success=False,
                error=deps_error,
                warnings=["Install required dependencies to run this experiment"]
            )

        try:
            # Load model
            self._load_model()

            # Define prefix
            prefix = (
                "The development of artificial intelligence has progressed through "
                "several important phases, starting with early theoretical work in the "
                "1950s and continuing through modern deep learning breakthroughs."
            )
            self.log(f"Using prefix: '{prefix[:50]}...'")

            # Get layers to analyze
            layers = self._get_layers_to_analyze()
            self.log(f"Analyzing layers: {layers}")

            # Generate continuations
            cfg = self.internal_config
            self.log(f"Generating {cfg.num_continuations} continuations...")

            continuations = self._generate_continuations_local(
                prefix=prefix,
                num_continuations=cfg.num_continuations,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature
            )
            self.log(f"Generated {len(continuations)} continuations")

            # Determine positions to sample
            positions_to_sample = [p for p in cfg.sample_positions if p <= cfg.max_tokens]
            if len(positions_to_sample) < 3:
                positions_to_sample = [10, 20, 40, 60, 80, 100]
                positions_to_sample = [p for p in positions_to_sample if p <= cfg.max_tokens]

            self.log(f"Sampling at positions: {positions_to_sample}")

            # Initialize storage for activations by state type, layer, and position
            # Structure: {state_type: {layer_idx: {position: [activations]}}}
            all_activations: Dict[str, Dict[int, Dict[int, List[np.ndarray]]]] = {}
            for state_type in cfg.state_types:
                all_activations[state_type] = {layer: {} for layer in layers}

            # Process each continuation
            for cont_idx, continuation in enumerate(continuations):
                words = continuation.split()

                for pos in positions_to_sample:
                    if pos <= len(words):
                        text = prefix + " " + " ".join(words[:pos])
                    else:
                        text = prefix + " " + continuation

                    # Extract internal states
                    layer_acts = self._extract_internal_states(text, layers)

                    # Store activations
                    for layer_idx, acts in layer_acts.items():
                        for state_type in cfg.state_types:
                            act = getattr(acts, state_type, None)
                            if act is not None:
                                if pos not in all_activations[state_type][layer_idx]:
                                    all_activations[state_type][layer_idx][pos] = []
                                all_activations[state_type][layer_idx][pos].append(act)

                if self.config.output.verbose and (cont_idx + 1) % 5 == 0:
                    self.log(f"Processed {cont_idx + 1}/{len(continuations)} continuations")

            # Analyze each state type and layer
            results_by_state = {}
            hurst_summary = {}

            for state_type in cfg.state_types:
                results_by_state[state_type] = {}
                hurst_values = []

                for layer_idx in layers:
                    activations_by_pos = all_activations[state_type][layer_idx]

                    if not activations_by_pos:
                        continue

                    positions, variances = self._compute_layer_variance(
                        activations_by_pos,
                        f"{state_type}_L{layer_idx}"
                    )

                    analysis = self._analyze_layer_dynamics(
                        layer_idx, state_type, positions, variances
                    )

                    results_by_state[state_type][layer_idx] = analysis

                    if analysis.get("valid", False):
                        hurst_values.append(analysis["hurst_exponent"])
                        self.log(
                            f"  {state_type} L{layer_idx}: "
                            f"H={analysis['hurst_exponent']:.3f}, "
                            f"R²={analysis['r_squared']:.3f}"
                        )

                if hurst_values:
                    hurst_summary[state_type] = {
                        "mean": float(np.mean(hurst_values)),
                        "std": float(np.std(hurst_values)),
                        "min": float(np.min(hurst_values)),
                        "max": float(np.max(hurst_values)),
                        "n_layers": len(hurst_values),
                    }

            # Compare pre/post LayerNorm
            ln_comparison = self._compare_layernorm_states(results_by_state)

            # Cross-layer consistency analysis
            cross_layer = self._analyze_cross_layer_consistency(results_by_state, layers)

            # Overall validation: check if internal states show similar H to embeddings
            overall_hurst_values = []
            for state_type in cfg.state_types:
                if state_type in hurst_summary:
                    overall_hurst_values.append(hurst_summary[state_type]["mean"])

            if overall_hurst_values:
                mean_hurst = np.mean(overall_hurst_values)
                h_tolerance = 0.25
                prediction_validated = abs(mean_hurst - 0.5) < h_tolerance
            else:
                mean_hurst = None
                prediction_validated = False

            self.log(f"\n=== Summary ===")
            self.log(f"Mean Hurst exponent across states: {mean_hurst:.3f}" if mean_hurst else "No valid results")
            self.log(f"Prediction (H~0.5): {'PASS' if prediction_validated else 'FAIL'}")

            self.result = ExperimentResult(
                experiment_name=self.experiment_name,
                success=True,
                metrics={
                    "mean_hurst_exponent": float(mean_hurst) if mean_hurst else None,
                    "prediction_validated": prediction_validated,
                    "hurst_by_state_type": hurst_summary,
                    "layernorm_comparison": ln_comparison,
                    "cross_layer_consistency": cross_layer,
                },
                data={
                    "model_name": self.internal_config.model_name,
                    "num_layers_analyzed": len(layers),
                    "layers_analyzed": layers,
                    "state_types": cfg.state_types,
                    "num_continuations": len(continuations),
                    "positions_sampled": positions_to_sample,
                    "results_by_state": results_by_state,
                    "prefix": prefix,
                }
            )
            return self.result

        except Exception as e:
            self.log(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            self.result = ExperimentResult(
                experiment_name=self.experiment_name,
                success=False,
                error=str(e)
            )
            return self.result

        finally:
            # Clean up
            self._clear_hooks()
            if self._model is not None and torch is not None:
                del self._model
                self._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _compare_layernorm_states(
        self,
        results_by_state: Dict[str, Dict[int, Dict]]
    ) -> Dict[str, Any]:
        """Compare Hurst exponents between pre and post LayerNorm states."""
        pre_hurst = []
        post_hurst = []

        if "residual_pre" in results_by_state:
            for layer_data in results_by_state["residual_pre"].values():
                if layer_data.get("valid", False):
                    pre_hurst.append(layer_data["hurst_exponent"])

        if "residual_post" in results_by_state:
            for layer_data in results_by_state["residual_post"].values():
                if layer_data.get("valid", False):
                    post_hurst.append(layer_data["hurst_exponent"])

        if not pre_hurst or not post_hurst:
            return {"valid": False, "error": "Insufficient data for comparison"}

        return {
            "valid": True,
            "pre_layernorm_mean": float(np.mean(pre_hurst)),
            "pre_layernorm_std": float(np.std(pre_hurst)),
            "post_layernorm_mean": float(np.mean(post_hurst)),
            "post_layernorm_std": float(np.std(post_hurst)),
            "difference": float(np.mean(post_hurst) - np.mean(pre_hurst)),
            "consistent": abs(np.mean(post_hurst) - np.mean(pre_hurst)) < 0.1,
        }

    def _analyze_cross_layer_consistency(
        self,
        results_by_state: Dict[str, Dict[int, Dict]],
        layers: List[int]
    ) -> Dict[str, Any]:
        """Analyze consistency of Hurst exponent across layers."""
        layer_hurst = {layer: [] for layer in layers}

        for state_type, layer_results in results_by_state.items():
            for layer_idx, data in layer_results.items():
                if data.get("valid", False):
                    layer_hurst[layer_idx].append(data["hurst_exponent"])

        # Compute mean H per layer
        layer_means = {}
        for layer, h_values in layer_hurst.items():
            if h_values:
                layer_means[layer] = float(np.mean(h_values))

        if len(layer_means) < 2:
            return {"valid": False, "error": "Not enough layers with valid data"}

        means = list(layer_means.values())

        return {
            "valid": True,
            "layer_hurst_means": layer_means,
            "overall_mean": float(np.mean(means)),
            "overall_std": float(np.std(means)),
            "range": float(np.max(means) - np.min(means)),
            "consistent": np.std(means) < 0.1,
        }

    def plot(self, save_path: Optional[str] = None) -> None:
        """Generate plots for internal states analysis."""
        if self.result is None or not self.result.success:
            self.log("Cannot plot: no successful experiment result")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.log("matplotlib not available for plotting")
            return

        results_by_state = self.result.data.get("results_by_state", {})
        layers = self.result.data.get("layers_analyzed", [])
        state_types = self.result.data.get("state_types", [])

        # Create subplot grid
        n_states = len(state_types)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))

        for ax_idx, state_type in enumerate(state_types[:4]):
            ax = axes[ax_idx]

            if state_type not in results_by_state:
                ax.set_title(f"{state_type}: No data")
                continue

            state_results = results_by_state[state_type]

            for layer_idx, color in zip(layers, colors):
                if layer_idx not in state_results:
                    continue

                data = state_results[layer_idx]
                if not data.get("valid", False):
                    continue

                positions = np.array(data["positions"])
                variances = np.array(data["variances"])
                hurst = data["hurst_exponent"]

                # Plot data points
                ax.scatter(positions, variances, color=color, alpha=0.7, s=50,
                          label=f"L{layer_idx} (H={hurst:.2f})")

                # Fitted line
                pos_smooth = np.linspace(positions.min(), positions.max(), 50)
                fitted = data["amplitude"] * np.power(pos_smooth, 2 * hurst)
                ax.plot(pos_smooth, fitted, color=color, alpha=0.5, linestyle="--")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Position (tokens)", fontsize=10)
            ax.set_ylabel("Variance", fontsize=10)
            ax.set_title(f"{state_type}", fontsize=12)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Add overall title with summary
        hurst_summary = self.result.metrics.get("hurst_by_state_type", {})
        summary_text = "Mean H: " + ", ".join(
            f"{st}: {data['mean']:.2f}"
            for st, data in hurst_summary.items()
        )
        fig.suptitle(
            f"Internal States Variance Growth\n{summary_text}",
            fontsize=14
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            self.log(f"Plot saved to {save_path}")
        elif self.config.output.show_plots:
            plt.show()

        plt.close(fig)

        # Additional plot: Cross-layer Hurst comparison
        self._plot_cross_layer_comparison(save_path)

    def _plot_cross_layer_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot Hurst exponent comparison across layers."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        results_by_state = self.result.data.get("results_by_state", {})
        layers = self.result.data.get("layers_analyzed", [])
        state_types = self.result.data.get("state_types", [])

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(layers))
        width = 0.2

        for i, state_type in enumerate(state_types[:4]):
            if state_type not in results_by_state:
                continue

            hurst_values = []
            for layer in layers:
                if layer in results_by_state[state_type]:
                    data = results_by_state[state_type][layer]
                    if data.get("valid", False):
                        hurst_values.append(data["hurst_exponent"])
                    else:
                        hurst_values.append(np.nan)
                else:
                    hurst_values.append(np.nan)

            ax.bar(x + i * width, hurst_values, width, label=state_type, alpha=0.8)

        # Reference line at H=0.5
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="H=0.5 (Brownian)")

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Hurst Exponent", fontsize=12)
        ax.set_title("Hurst Exponent by Layer and State Type", fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            # Modify path for second plot
            base, ext = save_path.rsplit(".", 1) if "." in save_path else (save_path, "png")
            cross_layer_path = f"{base}_cross_layer.{ext}"
            fig.savefig(cross_layer_path, dpi=150, bbox_inches="tight")
            self.log(f"Cross-layer plot saved to {cross_layer_path}")
        elif self.config.output.show_plots:
            plt.show()

        plt.close(fig)
