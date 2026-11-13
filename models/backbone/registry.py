# factory backbone

import inspect

def get_backbone(args: object):
    from models.backbone.vit1d import ViT1DEncoder
    from models.backbone.cnn import DilatedCNNEncoder, CNNEncoder

    BACKBONE_REGISTRY = {
        "vit1d": ViT1DEncoder,
        "dilated_cnn": DilatedCNNEncoder,
        "cnn":CNNEncoder,
    }

    backbone_name = args.model
    backbone_domain = getattr(args, "domain", "time")  # Default to "time" if not specified
    backbone_scaling_method = getattr(args, "scaling_method", None)  # Default to None if not specified

    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Backbone '{backbone_name}' is not registered. Available backbones: {list(BACKBONE_REGISTRY.keys())}")
    else :
        print(f"Using backbone: {backbone_name} with domain: {backbone_domain} and scaling method: {backbone_scaling_method}")
        backbone_class = BACKBONE_REGISTRY[backbone_name]

    # Convert args -> dict
    args_dict = vars(args)

    # Inspecte la signature du constructeur
    sig = inspect.signature(backbone_class.__init__)
    valid_args = {
        k: v for k, v in args_dict.items()
        if k in sig.parameters and k != "self"
    }

    # Ajouter domain et normalization s'ils sont acceptés par le constructeur
    if "domain" in sig.parameters:
        valid_args["domain"] = backbone_domain

    if "scaling_method" in sig.parameters:
        valid_args["scaling_method"] = backbone_scaling_method

    backbone = backbone_class(**valid_args)

    return backbone
