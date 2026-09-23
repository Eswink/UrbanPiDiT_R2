"""模型子模块（UrbanPiDiT V5.3.1 MorphoProcessDiT）。"""

try:
    from .anisotropic_urban_process_graph import AnisotropicUrbanProcessGraph
    from .micromet_coupling import MicroMetCouplingOperator
    from .model_registry import (
        build_model_from_config,
        build_model_v53_from_config,
        build_model_v531_from_config,
        dropped_model_kwargs,
        filter_model_kwargs,
    )
    from .morpho_process_proxy import MorphologyProcessProxyEncoder, MorphologyProcessProxyOutput
    from .morphology_graph import HeterogeneousMorphologyGraph, WindAwareMorphologyGraph
    from .process_conditioned_adaln import ProcessConditionedAdaLN
    from .static_encoder import StaticFeatureEncoder, StaticMorphologyEncoder
    from .urban_canopy import UrbanCanopyCoupling
    from .urban_morphology_control import UrbanMorphologyControlBranch
    from .variable_graph_v2 import ProxyConditionedVariableGraph
    from .urban_pidit import AdaptiveWeightNet, ModelConfig, UrbanPiDiT, build_model
except ImportError:
    from models.anisotropic_urban_process_graph import AnisotropicUrbanProcessGraph
    from models.micromet_coupling import MicroMetCouplingOperator
    from models.model_registry import (
        build_model_from_config,
        build_model_v53_from_config,
        build_model_v531_from_config,
        dropped_model_kwargs,
        filter_model_kwargs,
    )
    from models.morpho_process_proxy import MorphologyProcessProxyEncoder, MorphologyProcessProxyOutput
    from models.morphology_graph import HeterogeneousMorphologyGraph, WindAwareMorphologyGraph
    from models.process_conditioned_adaln import ProcessConditionedAdaLN
    from models.static_encoder import StaticFeatureEncoder, StaticMorphologyEncoder
    from models.urban_canopy import UrbanCanopyCoupling
    from models.urban_morphology_control import UrbanMorphologyControlBranch
    from models.variable_graph_v2 import ProxyConditionedVariableGraph
    from models.urban_pidit import AdaptiveWeightNet, ModelConfig, UrbanPiDiT, build_model

__all__ = [
    "UrbanPiDiT",
    "ModelConfig",
    "AdaptiveWeightNet",
    "AnisotropicUrbanProcessGraph",
    "MicroMetCouplingOperator",
    "MorphologyProcessProxyEncoder",
    "MorphologyProcessProxyOutput",
    "ProcessConditionedAdaLN",
    "UrbanMorphologyControlBranch",
    "StaticFeatureEncoder",
    "StaticMorphologyEncoder",
    "HeterogeneousMorphologyGraph",
    "WindAwareMorphologyGraph",
    "UrbanCanopyCoupling",
    "ProxyConditionedVariableGraph",
    "build_model",
    "build_model_from_config",
    "build_model_v53_from_config",
    "build_model_v531_from_config",
    "dropped_model_kwargs",
    "filter_model_kwargs",
]
