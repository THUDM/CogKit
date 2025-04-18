from cogkit import GenerationMode


def get_resolutions(task: GenerationMode) -> list[str]:
    if task == GenerationMode.TextToImage:
        return [
            "512x512",
            "512x768",
            "512x1024",
            "720x1280",
            "768x768",
            "1024x1024",
            "1080x1920",
        ]
    elif task == GenerationMode.TextToVideo:
        return [
            "49x480x720",
            "81x768x1360",
        ]
