def get_model(model_name, args):
    name = model_name.lower()
    if name == "tagfex":
        from models.tagfex import TagFex
        return TagFex(args)

    raise NotImplementedError(f"Standalone TagFex package only supports model: {model_name}")
